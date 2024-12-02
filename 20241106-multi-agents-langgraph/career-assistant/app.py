import os

import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from streamlit_chat import message
from streamlit_pills import pills

from agents import define_graph
from st_callable_util import get_streamlit_cb

load_dotenv(override=True)

# Set environment variables from Streamlit secrets or .env
os.environ["LINKEDIN_EMAIL"] = st.secrets.get("LINKEDIN_EMAIL", "")
os.environ["LINKEDIN_PASS"] = st.secrets.get("LINKEDIN_PASS", "")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") or st.secrets.get("LANGCHAIN_TRACING_V2", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "")
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
os.environ["SERPER_API_KEY"] = st.secrets.get("SERPER_API_KEY", "")
os.environ["FIRECRAWL_API_KEY"] = st.secrets.get("FIRECRAWL_API_KEY", "")
os.environ["LINKEDIN_SEARCH"] = st.secrets.get("LINKEDIN_JOB_SEARCH", "")
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets.get("AZURE_OPENAI_API_KEY", "")

# Page configuration
st.set_page_config(layout="wide")
st.title("GenAI Career Assistant - 👨‍💼")

streamlit_analytics.start_tracking()

# Setup directories and paths
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Sidebar - File Upload
uploaded_document = st.sidebar.file_uploader("Upload Your Resume", type="pdf")
if uploaded_document:
    bytes_data = uploaded_document.read()
    filepath = os.path.join(temp_dir, "resume.pdf")

    with open(filepath, "wb") as f:
        f.write(bytes_data)

    st.sidebar.markdown("Resume uploaded successfully! 📄")

st.markdown("**Make sure you have uploaded the resume!**")

# Sidebar - Service Provider Selection
service_provider = st.sidebar.selectbox(
    "Service Provider",
    ("openai", "azure-openai"),
)
streamlit_analytics.stop_tracking()

# Not to track the key
if service_provider == "openai":
    # Sidebar - OpenAI Configuration
    api_key_openai = st.sidebar.text_input(
        "OpenAI API Key",
        os.environ.get("OPENAI_API_KEY", ""),
        type="password",
    )
    model_openai = st.sidebar.selectbox(
        "OpenAI Model",
        ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
    )
    settings = {
        "model": model_openai,
        "model_provider": "openai",
        "temperature": 0.3,
    }
    st.session_state["OPENAI_API_KEY"] = api_key_openai
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]

elif service_provider == "azure-openai":
    azure_openai_endpoint = st.sidebar.text_input(
        "Azure OpenAI API Endpoint",
        os.environ.get("AZURE_OPENAI_ENDPOINT", "https://australiaeast.api.cognitive.microsoft.com"),
    )
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_openai_endpoint

    azure_openai_key = st.sidebar.text_input(
        "Azure OpenAI API Key",
        os.environ.get("AZURE_OPENAI_API_KEY"),
        type="password",
    )
    os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_key

    os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"

    model_azure_openai = st.sidebar.selectbox(
        "Azure OpenAI Model",
        ("gpt-4o", "gpt-4", "gpt-35-turbo"),
    )
    settings = {
        "model_provider": "azure_openai",
        "model": model_azure_openai,
        "temperature": 0.3,
    }

else:
    # Toggle visibility for Groq API Key input
    if "groq_key_visible" not in st.session_state:
        st.session_state["groq_key_visible"] = False

    if st.sidebar.button("Enter Groq API Key (optional)"):
        st.session_state["groq_key_visible"] = True

    if st.session_state["groq_key_visible"]:
        api_key_groq = st.sidebar.text_input("Groq API Key", type="password")
        st.session_state["GROQ_API_KEY"] = api_key_groq
        os.environ["GROQ_API_KEY"] = api_key_groq

    settings = {
        "model": "llama-3.1-70b-versatile",
        "model_provider": "groq",
        "temperature": 0.3,
    }

# Create the agent flow
flow_graph = define_graph()
message_history = StreamlitChatMessageHistory()

# Initialize session state variables
if "active_option_index" not in st.session_state:
    st.session_state["active_option_index"] = None
if "interaction_history" not in st.session_state:
    st.session_state["interaction_history"] = []
if "response_history" not in st.session_state:
    st.session_state["response_history"] = ["Hello! How can I assist you today?"]
if "user_query_history" not in st.session_state:
    st.session_state["user_query_history"] = ["Hi there! 👋"]

# Containers for the chat interface
conversation_container = st.container(border=True)
input_section = st.container()


def execute_chat_conversation(user_input, graph):
    st_callback = get_streamlit_cb(st.container(border=True))

    try:
        output = graph.invoke(
            {
                "messages": list(message_history.messages) + [user_input],
                "user_input": user_input,
                "config": settings,
                "callback": st_callback,
            },
            {"recursion_limit": 30},
        )
        message_output = output.get("messages")[-1]
        messages_list = output.get("messages")
        message_history.clear()
        message_history.add_messages(messages_list)

    except Exception as exc:
        print(exc)
        return ":( Sorry, Some error occurred. Can you please try again?"
    return message_output.content


# Clear Chat functionality
if st.button("Clear Chat"):
    st.session_state["user_query_history"] = []
    st.session_state["response_history"] = []
    message_history.clear()
    st.rerun()  # Refresh the app to reflect the cleared chat

# for tracking the query.
streamlit_analytics.start_tracking()

# Display chat interface
with input_section:
    options = [
        "Summarize my resume",
        "Analyze my resume and suggest a suitable full time job role and search for relevant job listings in New Zealand",
        "Identify top trends in the tech industry relevant to gen ai",
        "Find emerging technologies and their potential impact on job opportunities",
        "Create a career path visualization based on my skills and interests from my resume",
        "Job Search GenAI jobs in New Zealand.",
        "Generate a cover letter for my resume.",
    ]
    icons = ["📝", "🔍", "📈", "🌐", "💡", "🇳🇿", "✉️"]

    selected_query = pills(
        "Pick a question for query:",
        options,
        clearable=None,  # type: ignore
        icons=icons,
        index=st.session_state["active_option_index"],
        key="pills",
    )
    if selected_query:
        st.session_state["active_option_index"] = options.index(selected_query)

    # Display text input form
    with st.form(key="query_form", clear_on_submit=True):
        user_input_query = st.text_input(
            "Query:",
            value=(selected_query if selected_query else "Create a career path visualization based on my skills and interests from my resume"),
            placeholder="📝 Write your query or select from the above",
            key="input",
        )
        submit_query_button = st.form_submit_button(label="Send")

    if submit_query_button:
        if service_provider == "openai" and not st.session_state["OPENAI_API_KEY"]:
            st.error("Please enter your OpenAI API key before submitting a query.")

        elif service_provider == "azure-openai" and not os.environ.get("AZURE_OPENAI_API_KEY"):
            st.error("Please enter your Azure OpenAI API key before submitting a query.")

        elif user_input_query:
            # Process the query as usual if resume is uploaded
            chat_output = execute_chat_conversation(user_input_query, flow_graph)
            st.session_state["user_query_history"].append(user_input_query)
            st.session_state["response_history"].append(chat_output)
            st.session_state["last_input"] = user_input_query  # Save the latest input
            st.session_state["active_option_index"] = None

# Display chat history
if st.session_state["response_history"]:
    with conversation_container:
        for i in range(len(st.session_state["response_history"])):
            message(
                st.session_state["user_query_history"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="fun-emoji",
            )
            message(
                st.session_state["response_history"][i],
                key=str(i),
                avatar_style="bottts",
            )

streamlit_analytics.stop_tracking()
