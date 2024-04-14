import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import re
import google.generativeai as genai
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()
import PyPDF2
import io
# Set up Azure Text Analytics API
text_analytics_key = os.environ.get("YOUR_TEXT_ANALYTICS_KEY")
text_analytics_endpoint = os.environ.get("YOUR_TEXT_ANALYTICS_ENDPOINT")
text_analytics_client = TextAnalyticsClient(endpoint=text_analytics_endpoint, credential=AzureKeyCredential(text_analytics_key))

# Set up Gemini AI API key
gemini_api_key = os.environ.get("YOUR_GEMINI_KEY")
os.environ['GOOGLE_API_KEY'] = gemini_api_key

# Configure the Generative AI API
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')
if "rules" not in st.session_state:
    st.session_state.rules = [
    {
        "description": "Block personal information",
        "pattern": r"(?i)\b(ssn|social security number|credit card number|bank account number)\b"
    },
    {
        "description": "Block violent language",
        "pattern": r"(?i)\b(kill|murder|violence|violent)\b"
    },
    {
    "description": "Block personal or copyrighted information",
    "pattern": r"(?i)\b(email address|phone number|address|password|copyrighted material)\b"
    },
    {
    "description": "Block violent or self-harming intentions",
    "pattern": r"(?i)\b(suicide|self-harm|harm others|kill myself)\b"
    },
    {
    "description": "Block indications of illegal activities",
    "pattern": r"(?i)\b(insider trading|recruitment for illegal activities|malware development|data manipulation|data deletion)\b"
    },
    {
    "description": "Block providing copyrighted or personal information",
    "pattern": r"(?i)\b(sensitive document|personal identification|copyrighted material)\b"
    },
    {
    "description": "Block generating data relevant to illegal activities or hallucinations",
    "pattern": r"(?i)\b(insider trading data|hallucination inducing content)\b"
    },
    {
    "description": "Block prompts that could lead to malicious activities",
    "pattern": r"(?i)\b(injecting malware|promoting phishing|encouraging hacking)\b"
    }
]

if "rule_risk_levels" not in st.session_state:
    st.session_state.rule_risk_levels = {}

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_bytes):
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to generate rules from documents
def generate_rules(documents):
    rules = []
    for document in documents:
        if document.name.endswith('.pdf'):
            document_bytes = document.read()
            document_text = extract_text_from_pdf(document_bytes)
        else:
            document_text = document.read().decode("utf-8", errors="ignore")
        entities = text_analytics_client.recognize_entities([document_text])[0]
        for entity in entities.entities:
            if entity.category == "PersonName" or entity.category == "PersonalIdentifiableInformation":
                rule = {
                    "description": f"Block {entity.category.lower()}",
                    "pattern": rf"(?i)\b{entity.text}\b"
                }
                rules.append(rule)
    return rules

# Upload documents for rule generation
uploaded_files = st.file_uploader("Upload documents for rule generation", accept_multiple_files=True)
if uploaded_files:
    generated_rules = generate_rules(uploaded_files)
    st.write("Generated Rules:")
    for rule in generated_rules:
        st.write(f"- {rule['description']}")
    st.session_state.rules.extend(generated_rules)

# Function to check rules
def check_rules(text):
    for rule in st.session_state.rules:
        if re.search(rule["pattern"], text):
            return rule["description"]
    return None
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to handle user input and generate AI response
def chat_with_ai(user_input):
    # Check rules for user input
    query_rule_violation = check_rules(user_input)
    if query_rule_violation:
        st.error(f"Query blocked due to rule violation: {query_rule_violation}")
        return

    # Generate AI response
    response = model.generate_content(user_input)

    # Check if any candidates were returned
    if not response.candidates:
        st.error("Response blocked due to rule violation.")
        return

    # Get the generated text from the response
    generated_text = response.text

    # Check rules for AI response
    response_rule_violation = check_rules(generated_text)
    if response_rule_violation:
        response_text = f"Response blocked due to rule violation: {response_rule_violation}"
    else:
        response_text = generated_text

    # Store messages in chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response_text})
# Risk levels for rules

# Title and Description
st.title("AI Ethics Guardian Dashboard")
st.markdown("Welcome to the AI Ethics Guardian Dashboard! This dashboard provides insights into monitoring and controlling AI interactions.")
# Sidebar
st.sidebar.title("Settings")
selected_option = st.sidebar.selectbox("Select an Option", ["Overview", "Monitoring Rules", "Risk Analysis", "AI Interaction"])
# Sample Data
data = pd.DataFrame({
    'User ID': [1, 2, 3, 4, 5],
    'Query': ['Personal information request', 'Violent intentions expressed', 'Normal query', 'Data manipulation attempt', 'Sensitive information request'],
    'Risk Level': ['High', 'High', 'Low', 'Medium', 'High']
})
# Features based on selected option
if selected_option == "Overview":
    st.subheader("Overview")
    st.write("This section provides an overview of AI interactions and risk levels.")
    # Display sample data
    st.dataframe(data)
elif selected_option == "Monitoring Rules":
    st.subheader("Monitoring Rules")
    st.write("This section allows administrators to view and manage monitoring rules.")
    # Display existing rules
    st.subheader("Existing Rules")
    for rule in st.session_state.rules:
        st.write(f"- {rule['description']} (Risk Level: {st.session_state.rule_risk_levels.get(rule['description'], 'Not set')})")
    # Add/Edit monitoring rules
    st.subheader("Add/Edit Monitoring Rule")
    new_rule_description = st.text_input("Rule Description")
    new_rule_pattern = st.text_input("Rule Pattern (Regular Expression)")
    new_rule_risk_level = st.selectbox("Risk Level", ["Low", "Medium", "High"])
    if st.button("Save Rule"):
        new_rule = {
            "description": new_rule_description,
            "pattern": new_rule_pattern
        }
        st.session_state.rules.append(new_rule)
        st.session_state.rule_risk_levels[new_rule_description] = new_rule_risk_level
        st.success("Rule saved successfully!")
elif selected_option == "Risk Analysis":
    st.subheader("Risk Analysis")
    st.write("This section provides insights into risk analysis based on AI interactions.")
    # Display risk analysis chart
    fig = px.bar(data, x='User ID', y='Risk Level', color='Risk Level', title='Risk Analysis')
    st.plotly_chart(fig)
elif selected_option == "AI Interaction":
    st.subheader("AI Interaction")
    st.write("This section allows you to interact with the AI chatbot and see real-time rule-based filtering.")
    # Set Google API key
    os.environ['GOOGLE_API_KEY'] = gemini_api_key
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    # Create the Model
    model = genai.GenerativeModel('gemini-pro')
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me anything"
            }
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.markdown(message["role"]):
            st.markdown(message["content"])
    # Process and store Query and Response
    def llm_function(query):
        query_rule_violation = check_rules(query)
        if query_rule_violation:
            st.error(f"Query blocked due to rule violation: {query_rule_violation}")
            return
        response = model.generate_content(query)
    # Check if any candidates were returned
        if not response.candidates:
            st.error("Response blocked due to rule violation.")
            return
        generated_text = response.text
    # Check rules for response
        response_rule_violation = check_rules(generated_text)
        if response_rule_violation:
            response_text = f"Response blocked due to rule violation: {response_rule_violation}"
        else:
            response_text = generated_text
    # Displaying the Assistant Message
        with st.markdown("assistant"):
            st.markdown(response_text)
    # Storing the User Message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
    # Storing the Assistant Message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.text
        })
    # Accept user input
    query = st.text_input("What is up?", key="user_input")
    # Calling the Function when Input is Provided
    if query:
        # Displaying the User Message
        with st.markdown("user"):
            st.markdown(query)
        llm_function(query)
# Footer
st.markdown("---")
st.markdown("Developed by AI Ethics Guardian Team")