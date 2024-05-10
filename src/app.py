import os
import streamlit as st
from model import ChatModel
import rag_util

FILES_DIR = "/content/drive/MyDrive/LLM_RAG_Bot/files"

# Set page configuration
st.set_page_config(
    page_title="Gemma 2B Chatbot",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styles
css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f5f5f5;
        }

        .stApp {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .stHeader {
            text-align: center;
            margin-bottom: 30px;
        }

        .stHeader h1 {
            font-weight: 700;
            color: #333333;
        }

        .stHeader p {
            color: #666666;
        }

        .stSidebar .stMarkdown {
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .stSidebar h2 {
            font-weight: 700;
            color: #333333;
            margin-bottom: 10px;
        }

        .stSidebar p {
            color: #666666;
            margin-bottom: 20px;
        }

        .stChat {
            margin-top: 30px;
        }

        .stChatMessage {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }

        .stChatMessage.user {
            background-color: #e6f3ff;
            color: #333333;
        }

        .stChatMessage.assistant {
            background-color: #f5f5f5;
            color: #333333;
        }
    </style>
"""

# HTML code for the header
header_html = """
    <div class="stHeader">
        <h1>Gemma 2B Chatbot</h1>
        <p>An AI-powered chatbot leveraging the Gemma 2B language model and Retrieval-Augmented Generation (RAG) for context-aware responses.</p>
    </div>
"""

# Render the header and custom CSS styles
st.markdown(css, unsafe_allow_html=True)
st.markdown(header_html, unsafe_allow_html=True)

# Load the models using cache for better performance
@st.cache_resource
def load_model():
    model = ChatModel(model_id="mustafaaljadery/gemma-2b-10m", device="cuda")
    return model

@st.cache_resource
def load_encoder():
    encoder = rag_util.Encoder(model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu")
    return encoder

model = load_model()
encoder = load_encoder()

# Helper function to save uploaded files
def save_file(uploaded_file):
    """Save uploaded PDF files to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Sidebar for user inputs
with st.sidebar:
    st.markdown("## Options")
    max_new_tokens = st.number_input(
        "Max Output Length",
        min_value=128,
        max_value=16384,
        value=2048,
        step=128,
        help="Maximum number of tokens to generate for the response.",
    )
    k = st.number_input(
        "Number of Retrieved Documents",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of top relevant documents to retrieve from the vector database.",
    )
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=['pdf', 'PDF'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to use as context for the chatbot.",
    )

# Process uploaded files and create vector database
file_paths = []
for file in uploaded_files:
    file_paths.append(save_file(file))

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        docs = rag_util.load_and_split_pdfs(file_paths)
        DB = rag_util.FaissDb(docs=docs, embedding_function=encoder.embedding_function)
    st.success("PDF files processed successfully!")
else:
    st.warning("Please upload at least one PDF file to provide context for the chatbot.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Accept user input and generate response
if prompt := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Display user message
    with st.chat_message('user'):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message('assistant'):
        user_prompt = st.session_state.messages[-1]['content']
        context = (None if not uploaded_files else DB.similarity_search(user_prompt, k=k))
        with st.spinner("Generating response..."):
            answer = model.generate(user_prompt, context=context, max_new_tokens=max_new_tokens)
        response = st.markdown(answer)
        st.session_state.messages.append({'role': 'assistant', 'content': answer})
