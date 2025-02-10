import streamlit as st
import requests

###############################################################################
# Default Named System Prompts
###############################################################################
default_system_prompts = {
    "Helpful AI": (
        "You are a helpful AI assistant. Provide useful, precise, and correct "
        "answers to user queries."
    ),
    "Creative Writer": (
        "You are a creative writer who loves storytelling. Provide imaginative "
        "narrative responses to user instructions."
    ),
    "Instructor": (
        "You are an instructor who explains topics succinctly and systematically."
    )
}

###############################################################################
# LLM Call Functions
###############################################################################

def call_openai_api(system_prompt, conversation_messages, model, temperature, max_tokens):
    api_url = "https://api.openai.com/v1/chat/completions"

    # Retrieve from st.secrets
    openai_api_key = st.secrets["openai_api_key"]

    messages = [{"role": "system", "content": system_prompt}] + conversation_messages

    data = {
        "model": model,  # e.g. "gpt-4o", "gpt-4o-mini"
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        completion = response.json()
        return completion["choices"][0]["message"]["content"]
    except (requests.exceptions.RequestException, KeyError) as e:
        return f"Error calling ChatGPT API: {e}"


def call_groq_api(system_prompt, conversation_messages, model, temperature, max_tokens):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    groq_api_key = st.secrets["groq_api_key"]
    messages = [{"role": "system", "content": system_prompt}] + conversation_messages
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_api_key}"
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        completion = response.json()
        return completion["choices"][0]["message"]["content"]
    except (requests.exceptions.RequestException, KeyError) as e:
        return f"Error calling Groq API: {e}"

def call_google_api(system_prompt, conversation_messages, model, temperature, max_tokens):
    google_api_key = st.secrets["google_api_key"]  # "GEMINI_API_KEY" from secrets
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    endpoint = f"{base_url}/models/{model}:generateContent?key={google_api_key}"

    # Combine conversation for demonstration only. You can adjust to your strategy.
    combined_text = system_prompt + "\n"
    for msg in conversation_messages:
        if msg["role"] == "user":
            combined_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            combined_text += f"Assistant: {msg['content']}\n"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": combined_text.strip()}
                ]
            }
        ]
        # Optionally specify "temperature", "maxOutputTokens" or other fields if your endpoint supports them
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        reply = response.json()
        text_response = reply["candidates"][0]["content"]["parts"][0]["text"]
        return text_response

    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        return f"Error calling Google API: {e}"
###############################################################################
# Streamlit Multi-turn Conversation App with Bottomsheet Chat
###############################################################################

def main():
    st.title("LLM Chatbot")

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "system_prompt_text" not in st.session_state:
        # Default to "Helpful AI" from the start
        st.session_state["system_prompt_text"] = default_system_prompts["Helpful AI"]

    if "custom_system_prompts" not in st.session_state:
        st.session_state["custom_system_prompts"] = {}

    # Sidebar: LLM Provider & Model
    st.sidebar.header("LLM Configuration")
    provider = st.sidebar.selectbox(
        "LLM API Provider",
        ["Groq", "ChatGPT", "Google"]  # Added "Google" here
    )

    if provider == "Groq":
        model = st.sidebar.selectbox(
            "Choose model",
            ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "deepseek-r1-distill-llama-70b","mixtral-8x7b-32768"],
            index=0
        )
        
    elif provider == "ChatGPT":
        model = st.sidebar.selectbox(
            "Choose model",
            ["gpt-4o-mini","gpt-4o", ],
            index=0
        )
    else:  # Google
        # For example "gemini-1.5-flash", or any other model name if you have them available
        model = st.sidebar.selectbox(
            "Choose model",
            ["gemini-1.5-flash","gemini-2.0-flash-thinking-exp-01-21","gemini-2.0-pro-exp-02-05"],
            index=0
        )

    # Sidebar: System Prompt
    st.sidebar.header("System Prompt")
    combined_prompt_names = (
        list(default_system_prompts.keys())
        + list(st.session_state["custom_system_prompts"].keys())
        + ["Add New System Prompt"]
    )

    selected_prompt_name = st.sidebar.selectbox(
        "Select Prompt Name:",
        combined_prompt_names
    )

    if selected_prompt_name == "Add New System Prompt":
        with st.sidebar.expander("Add New Prompt"):
            new_prompt_name = st.text_input("Enter custom prompt name:")
            new_prompt_text = st.text_area("Enter full system prompt text:")
            if st.button("Save New Prompt"):
                if new_prompt_name.strip() and new_prompt_text.strip():
                    st.session_state["custom_system_prompts"][new_prompt_name] = new_prompt_text
                    st.session_state["system_prompt_text"] = new_prompt_text
                    st.success(f"Added new prompt: {new_prompt_name}")
                else:
                    st.error("Please provide both a name and text for the new system prompt.")
    else:
        if selected_prompt_name in default_system_prompts:
            st.session_state["system_prompt_text"] = default_system_prompts[selected_prompt_name]
        elif selected_prompt_name in st.session_state["custom_system_prompts"]:
            st.session_state["system_prompt_text"] = st.session_state["custom_system_prompts"][selected_prompt_name]

    # LLM Parameters + Reset
    st.sidebar.header("LLM Parameter Tweaking")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 1, 2048, 128, 1)

    if st.sidebar.button("Reset context"):
        st.session_state["messages"] = []

    # Display Conversation
    for msg in st.session_state["messages"]:
        if msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("user"):
                st.markdown(msg["content"])

    # Bottomsheet Chat Input
    if user_text := st.chat_input("Your message"):
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.spinner("Thinking..."):
            # Call the correct provider function
            if provider == "ChatGPT":
                response_text = call_openai_api(
                    system_prompt=st.session_state["system_prompt_text"],
                    conversation_messages=st.session_state["messages"],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif provider == "Groq":
                response_text = call_groq_api(
                    system_prompt=st.session_state["system_prompt_text"],
                    conversation_messages=st.session_state["messages"],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:  # Google
                response_text = call_google_api(
                    system_prompt=st.session_state["system_prompt_text"],
                    conversation_messages=st.session_state["messages"],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

        st.session_state["messages"].append({"role": "assistant", "content": response_text})

        with st.chat_message("assistant"):
            st.markdown(response_text)


if __name__ == "__main__":
    main()
