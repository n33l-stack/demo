import os
import asyncio
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_nomic import NomicEmbeddings
from langchain_groq import ChatGroq
from langchain_community.retrievers.pinecone_hybrid_search import PineconeHybridSearchRetriever
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from functools import lru_cache

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(
    page_title="StanBot - UofT IC Help Desk",
    layout="wide",
    page_icon="👨‍💻",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Sophie"

# Model configurations
MODEL_CONFIGS = {
    "Sophie": "llama-3.3-70b-specdec",
    "Nolan": "deepseek-r1-distill-llama-70b",
    "Daniel": "gemma2-9b-it",
    "Saba": "mistral-saba-24b"
}

@lru_cache(maxsize=1)
def get_pinecone_client():
    """Cache Pinecone client to avoid repeated initialization"""
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])

@lru_cache(maxsize=1)
def get_bm25_encoder():
    """Cache BM25 encoder to avoid repeated loading"""
    return BM25Encoder().load("bm25_values.json")

@lru_cache(maxsize=1)
def get_embeddings():
    """Cache embeddings model to avoid repeated initialization"""
    return NomicEmbeddings(
        nomic_api_key=os.environ['NOMIC_API_KEY'],
        model='nomic-embed-text-v1.5',
        dimensionality=256
    )

@lru_cache(maxsize=1)
def get_llm():
    """Cache LLM to avoid repeated initialization"""
    model_name = MODEL_CONFIGS[st.session_state.selected_model]
    return ChatGroq(
        temperature=0.5,
        model_name=model_name,
        api_key=os.environ["GROQ_API_KEY"],
        max_tokens=2048
    )

async def initialize_components():
    """Asynchronously set up components"""
    try:
        # Initialize components concurrently
        pc = get_pinecone_client()
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        
        # Get cached components
        bm25_encoder = get_bm25_encoder()
        embeddings = get_embeddings()
        
        # Initialize retriever
        retriever = PineconeHybridSearchRetriever(
            sparse_encoder=bm25_encoder,
            embeddings=embeddings,
            index=index
        )
        
        # Initialize LLM
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-specdec",
            temperature=0.5,
            max_tokens=2048
        )
        
        return retriever, llm
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

# Define RAG workflow
SYSTEM_PROMPT = """You are {assistant_name}, a UofT IT Help Desk assistant powered by the {model_name} language model. Use:
- {context} for verified knowledge base info
- {chat_history} for conversation context
- {question} for current query

When asked about your identity or name:
- Your name is {assistant_name}
- You run on the {model_name} model
- You are a UofT IT Help Desk assistant

Guidelines:
- Answer with verified info from knowledge base
- Use clear numbered steps for instructions
- Keep responses friendly but focused
- Avoid technical jargon unless necessary
"""

def optimize_chat_history(messages, max_messages=5):
    """
    Optimize chat history by keeping only the most recent and relevant messages
    - Keep last few message pairs (question-answer)
    - Limit total number of messages
    """
    # If we have 5 or fewer messages, return all
    if len(messages) <= max_messages:
        return [m.content for m in messages]
    
    # Keep the most recent messages
    recent_messages = messages[-max_messages:]
    return [m.content for m in recent_messages]

async def rag_function(state):
    """Async RAG processing function"""
    try:
        messages = state["messages"]
        if not messages or not isinstance(messages[-1], HumanMessage):
            st.error("No messages in state or last message is not a HumanMessage")
            return {"messages": messages}
        
        retriever, llm = await initialize_components()
        if not retriever or not llm:
            st.error("Failed to initialize components")
            return {"messages": messages + [AIMessage(content="Error: Could not initialize components")]}
        
        # Get the latest message
        query = messages[-1].content
        
        # Check if it's an identity question
        identity_questions = [
            "what is your name",
            "who are you",
            "what model are you",
            "which model are you",
            "what's your name",
            "what are you",
            "tell me about yourself",
            "introduce yourself",
            "what should I call you"
        ]
        
        is_identity_question = any(q in query.lower() for q in identity_questions)
        
        if is_identity_question:
            # For identity questions, we don't need to query the knowledge base
            context = ""
        else:
            # Get relevant documents for non-identity questions
            docs = await retriever.ainvoke(query)
            context = "\n".join(doc.page_content for doc in docs)
        
        # Optimize chat history to include only recent relevant messages
        chat_history = optimize_chat_history(messages[:-1])  # Exclude current query
        chat_history_str = "\n".join(chat_history)

        # Set up a structured prompt
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

        # Build final prompt message object list
        formatted_prompt = prompt_template.format_messages(
            context=context,
            chat_history=chat_history_str,
            question=query,
            assistant_name=st.session_state.selected_model,
            model_name=MODEL_CONFIGS[st.session_state.selected_model]
        )

        # Call the LLM with the structured message list
        response = await llm.ainvoke(formatted_prompt)
        
        # Create AIMessage from response
        ai_message = AIMessage(content=response.content)
        return {"messages": messages + [ai_message]}
        
    except Exception as e:
        st.error(f"Error in RAG function: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return {"messages": messages + [AIMessage(content="I apologize, but I encountered an error processing your request.")]}

# Define state type
class State(TypedDict):
    messages: list

# Configure LangGraph
workflow = StateGraph(State)
workflow.add_node("rag", rag_function)
workflow.set_entry_point("rag")
workflow.set_finish_point("rag")
app = workflow.compile(checkpointer=MemorySaver())

async def stream_response(response_text):
    """Asynchronously stream response"""
    message_placeholder = st.empty()
    full_response = ""
    
    sentences = response_text.split('. ')
    for sentence in sentences:
        if sentence:
            full_response += sentence + '. '
            message_placeholder.markdown(full_response + "▌")
            await asyncio.sleep(0.1)
    
    message_placeholder.markdown(full_response)
    return full_response

def render_sidebar():
    """Render sidebar content"""
    with st.sidebar:
        st.title("Information Commons Help Desk")
        
        # Model selector
        st.markdown("### 🤖 Choose your Assistant")
        selected_model = st.selectbox(
            "Select Model",
            options=list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS.keys()).index(st.session_state.selected_model),
            key="model_selector"
        )
        
        # Update selected model if changed
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            # Clear the LLM cache to force reinitialization with new model
            get_llm.cache_clear()
            # Add a rerun to update the interface
            st.rerun()
        
        st.markdown("---")
        
        st.header("📍 Location")
        st.write("Ground floor, Robarts Library\n130 St George St., Toronto")
        
        st.header("📞 Contact")
        st.write("Phone: 416-978-4357\nEmail: help.desk@utoronto.ca")
        
        st.header("⏰ Hours")
        st.write("Visit [help.ic.utoronto.ca](https://help.ic.utoronto.ca) for current hours")

def save_feedback(index):
    """Save feedback to the message history"""
    st.session_state.messages[index]["feedback"] = st.session_state[f"feedback_{index}"]

async def main():
    # Render sidebar
    render_sidebar()
    
    # Main chat interface title with animation
    st.markdown('<div class="title-container"><h2 class="title">StanBot - Your UofT IT Assistant 🤖</h2></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            # Add feedback widget for assistant messages
            if message["role"] == "assistant":
                feedback = message.get("feedback", None)
                st.session_state[f"feedback_{i}"] = feedback
                st.feedback(
                    "thumbs",
                    key=f"feedback_{i}",
                    on_change=save_feedback,
                    args=[i],
                )
    
    # Handle user input
    if prompt := st.chat_input("How can I help you with UofT IT services today?"):
        #st.info("Received user input")
        
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            try:
                # Convert messages to LangChain format
                messages = []
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "assistant":
                        messages.append(AIMessage(content=m["content"]))
                    else:
                        messages.append(HumanMessage(content=m["content"]))
                messages.append(HumanMessage(content=prompt))
                
                #st.info(f"Processing with {len(messages)} messages")
                
                # Set up config
                config = {
                    "configurable": {
                        "session_id": st.session_state.session_id,
                        "thread_id": st.session_state.thread_id,
                        "user_id": st.session_state.session_id
                    }
                }
                
                # Process message
                result = await app.ainvoke({"messages": messages}, config=config)
                #st.info(f"Got result: {result}")
                
                if "messages" in result and result["messages"]:
                    latest_msg = result["messages"][-1]
                    if isinstance(latest_msg, AIMessage):
                        #st.info("Streaming response...")
                        response = await stream_response(latest_msg.content)
                        if response:
                            # Add feedback widget for new response
                            st.feedback(
                                "thumbs",
                                key=f"feedback_{len(st.session_state.messages)}",
                                on_change=save_feedback,
                                args=[len(st.session_state.messages)]
                            )
                            # Add response to message history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })
                
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
