# ui.py
"""
Main UI page for the AI Cooking Assistant Chat.
"""

import requests
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Cooking Assistant",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- API Configuration ---
API_URL = "http://localhost:8000/query/assistant"

# --- UI Title ---
st.title("🍳 AI Cooking Companion")
st.caption("Your hands-free helper in the kitchen for troubleshooting and advice.")

# --- Session State Initialization for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you in the kitchen today?"}]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Example Prompts ---
st.sidebar.title("Not sure what to ask?")
st.sidebar.markdown("Try one of these prompts:")
if st.sidebar.button("My curry is too salty, what can I do?"):
    st.session_state.prompt_from_button = "My curry is too salty, what can I do?"
if st.sidebar.button("Give me a simple recipe for chicken breast."):
    st.session_state.prompt_from_button = "Give me a simple recipe for chicken breast."

# --- User Input Handling ---
prompt = st.chat_input("Ask a cooking question...")
if "prompt_from_button" in st.session_state:
    prompt = st.session_state.prompt_from_button
    del st.session_state.prompt_from_button

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            response = requests.post(API_URL, json={"query_text": prompt}, timeout=120)
            response.raise_for_status()
            api_response = response.json()
            full_response = api_response.get("response_text", "I'm sorry, I encountered an issue.")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except requests.exceptions.RequestException as e:
            error_message = f"I'm having trouble connecting to my brain... Please ensure the backend server is running. (Error: {e})"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})