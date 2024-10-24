from dotenv import load_dotenv # type: ignore
import streamlit as st
from langchain import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler
from streamlit_chat import message # type: ignore

st.set_page_config(page_title="GENERAL_CHATBOT_SYSTEM")

# Load environment variables if needed
load_dotenv()

# Initialize session state for conversation and chat history
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state or st.session_state.chat_history is None:
    st.session_state.chat_history = []  # Initialize as empty list

st.header("CHATBOT SYSTEM")

# Image or other general UI features
st.image("D:/myproject/Final_Project/Final_Bank_Project/ChatBot/chatbot1.png", caption="Ask anything!")

# Chat interface
response_container = st.container()

# Display the chat history if it exists
with response_container:
    if st.session_state.chat_history:
        for i, message_data in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(message_data['content'], is_user=True, key=str(i))
            else:
                message(message_data['content'], key=str(i))

# Define the chatbot conversation flow
def get_conversation_chain():
    
    handler = StdOutCallbackHandler()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.6, "max_length": 64})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    return llm , handler, memory # Simple LLM usage

def handle_user_input(user_question):
    # Run the model on the user input as a string prompt
    response = st.session_state.conversation(user_question)
    
    # Append question and response to chat history
    st.session_state.chat_history.append({'content': user_question, 'role': 'user'})
    st.session_state.chat_history.append({'content': response, 'role': 'bot'})

# Initialize the conversation chain
if st.session_state.conversation is None:
    st.session_state.conversation = get_conversation_chain()

# User input for questions
user_question = st.chat_input("Ask a question.")
if user_question:
    handle_user_input(user_question)
