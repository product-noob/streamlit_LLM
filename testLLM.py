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
    """
    Calls an OpenAI-like Chat Completion endpoint with every turn in `conversation_messages`.
    The first message is the system prompt. All subsequent messages are user or assistant roles.
    """
    api_url = "https://api.openai.com/v1/chat/completions"

    # Retrieve from st.secrets
    openai_api_key = st.secrets["openai_api_key"]

    messages = [{"role": "system", "content": system_prompt}] + conversation_messages

    data = {
        "model": model,
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
    """
    Calls the Groq endpoint for chat completions, passing in a system prompt
    as the first message, followed by the conversation.
    """
    api_url = "https://api.groq.com/openai/v1/chat/completions"

    # Retrieve from st.secrets
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


###############################################################################
# Main Streamlit App
###############################################################################

def main():
    st.title("Custom ChatGPT")

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "system_prompt_text" not in st.session_state:
        # Default to "Helpful AI" from the start
        st.session_state["system_prompt_text"] = default_system_prompts["Helpful AI"]

    if "custom_system_prompts" not in st.session_state:
        # Dictionary for user-added prompts: { prompt_name: prompt_text, ... }
        st.session_state["custom_system_prompts"] = {}

    # Sidebar: LLM Provider & Model
    st.sidebar.header("LLM Configuration")

    provider = st.sidebar.selectbox(
        "LLM API Provider",
        ["ChatGPT", "Groq"]  # Changed "OpenAI-like" to "ChatGPT"
    )

    if provider == "ChatGPT":
        model = st.sidebar.selectbox(
            "Choose model",
            ["gpt-4o", "gpt-4o-mini"],
            index=0
        )
    else:
        model = st.sidebar.selectbox(
            "Choose model",
            ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
            index=0
        )

    # Sidebar: System Prompt Selection (by name)
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

    # Sidebar: Parameters + Reset
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
        else:  # user
            with st.chat_message("user"):
                st.markdown(msg["content"])

    # Bottomsheet Chat Input
    if user_text := st.chat_input("Your message"):
        st.session_state["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.spinner("Thinking..."):
            if provider == "ChatGPT":
                response_text = call_openai_api(
                    system_prompt=st.session_state["system_prompt_text"],
                    conversation_messages=st.session_state["messages"],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response_text = call_groq_api(
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
