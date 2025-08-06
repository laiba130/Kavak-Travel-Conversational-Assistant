import streamlit as st
import os
from main import TravelAssistant

# Streamlit UI
def main():
    st.set_page_config(page_title="Kavak Travel Assistant", page_icon="✈️")
    st.title("✈️ Kavak Travel Assistant")
    st.markdown("Hello there! I'm here to help you with your travel plans. I can assist with flight searches, visa information, and airline policies.")
    st.markdown("Feel free to ask me anything about your travel needs.")
    
    # Get OpenAI API key from environment or user input
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        with st.sidebar:
            st.info("Please enter your OpenAI API key to get started.")
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            st.markdown("Your key is not stored and will only be used for this session.")
        
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()

    # Initialize the TravelAssistant in Streamlit's session state
    if "assistant" not in st.session_state:
        try:
            st.session_state.assistant = TravelAssistant(openai_api_key=openai_api_key)
        except Exception as e:
            st.error(f"Failed to initialize the travel assistant. Please check your API key. Error: {e}")
            st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a welcome message from the assistant
        st.session_state.messages.append(
            {"role": "assistant", "content": "How can I help with your travel plans today?"}
        )
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if user_query := st.chat_input("How can I help you today?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Get assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use a copy of the messages for the function call to avoid modifying the main state until the final response
                response_content = st.session_state.assistant.process_query(
                    user_query,
                    conversation_history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages if m["role"] in ["user", "assistant"]
                    ]
                )
                st.markdown(response_content)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()