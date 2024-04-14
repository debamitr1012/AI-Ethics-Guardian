from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import pdfplumber
import textwrap
import json
# Azure Text Analytics configuration
key = "YOUR_TEXT_ANALYTICS_KEY"
endpoint = "YOUR_TEXT_ANALYTICS_ENDPOINT"
credential = AzureKeyCredential(key)
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text
# Function to analyze text with Azure Text Analytics
def analyze_text(text):
    chunks = textwrap.wrap(text, 5120)  # Split text into chunks of 5120 characters
    key_phrases = []
    for chunk in chunks:
        key_phrases.extend(text_analytics_client.extract_key_phrases(documents=[chunk]))
    return key_phrases
# Function to generate rules
def generate_rules(text):
    # Example: Generate rules based on key phrases
    rules = []
    key_phrases = analyze_text(text)
    for phrase in key_phrases:
        rules.append(f"If '{phrase}' appears, do something...")
    return rules
def main():
    pdf_path = "D:/KnacktoHack/20230601STO93804_en.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    generated_rules = generate_rules(extracted_text)
    # Write generated rules to a JSON file
    with open("generated_rules.json", "w") as json_file:
        json.dump(generated_rules, json_file, indent=4)
    print("Generated rules saved to generated_rules.json")
if __name__ == "__main__":
    main()
