STYLES = """
<style>
    /* Global font */
    * {
        font-family: Arial, sans-serif !important;
    }
    
    /* Title styling */
    .title {
        text-align: center;
        overflow: hidden;
        white-space: nowrap;
        margin: 0 auto;
        display: inline-block;
        border-right: 2px solid #FFFFFF;
        animation: typing 3.5s steps(40, end), blink-caret 0.75s step-end infinite;
        color: #FFFFFF !important;
        max-width: fit-content;
    }
    
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #FFFFFF }
    }
    
    /* Title container */
    .title-container {
        width: 100%;
        text-align: center;
        display: flex;
        justify-content: center;
        margin-bottom: 1em;
    }
    
    /* Main background */
    .stApp {
        background-color: #00204E;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #003E74;
    }
    
    /* Chat containers */
    .stChatMessage {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 12px;
        margin: 6px 0;
        white-space: pre-wrap !important;
        overflow-wrap: break-word;
        word-wrap: break-word;
        line-height: 1.4;
    }
    
    /* Message text formatting */
    .stChatMessage p {
        margin-bottom: 0.7em;
        white-space: pre-wrap !important;
        color: #00204E;
    }
    
    .stChatMessage p:last-child {
        margin-bottom: 0;
    }
    
    /* User message specific */
    .stChatMessage [data-testid="StyledLinkIconContainer"] {
        background-color: #003E74;
    }
    
    /* Assistant message specific */
    .stChatMessage [data-testid="StyledLinkIconContainer"] + div {
        background-color: #FFFFFF;
    }
    
    /* Code blocks in messages */
    .stChatMessage code {
        background-color: #f0f0f0;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-size: 85%;
        font-family: monospace;
    }
    
    /* Pre blocks for multi-line code */
    .stChatMessage pre {
        background-color: #f0f0f0;
        padding: 0.8em;
        border-radius: 8px;
        overflow-x: auto;
        margin: 0.7em 0;
    }
    
    /* Chat input box */
    .stChatInputContainer {
        border-color: #003E74;
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 8px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        margin-bottom: 0.4em;
    }
    
    /* Links */
    a {
        color: #FFFFFF !important;
        text-decoration: underline;
    }
    
    /* Divider */
    hr {
        border-color: #003E74;
        margin: 0.7em 0;
    }
    
    /* Sidebar text */
    .sidebar .sidebar-content {
        color: #FFFFFF;
    }
    
    /* Model selector styling */
    .stSelectbox {
        background-color: #FFFFFF;
        border-radius: 8px;
        margin-bottom: 1em;
    }
    
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        color: #00204E;
        border: 1px solid #003E74;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #FFFFFF;
        border-radius: 8px;
    }
    
    /* Model selector label */
    .stSelectbox label {
        color: #FFFFFF !important;
    }
    
    /* Lists in messages */
    .stChatMessage ul, .stChatMessage ol {
        margin: 0.5em 0;
        padding-left: 1.5em;
    }
    
    .stChatMessage li {
        margin: 0.3em 0;
        color: #00204E;
    }

    /* Additional styles for better contrast */
    .stMarkdown {
        color: #FFFFFF;
    }
    
    .stTextInput > div > div > input {
        color: #00204E !important;
    }
    
    /* Feedback buttons styling */
    .stFeedback {
        margin-top: 0.5em;
    }
    
    .stFeedback button {
        background-color: transparent !important;
        border: 1px solid #003E74 !important;
        color: #FFFFFF !important;
        padding: 0.3em 0.6em !important;
        margin: 0 0.2em !important;
        border-radius: 4px !important;
    }
    
    .stFeedback button:hover {
        background-color: #003E74 !important;
        border-color: #FFFFFF !important;
    }
    
    .stFeedback button[data-selected="true"] {
        background-color: #003E74 !important;
        border-color: #FFFFFF !important;
    }
</style>
""" 