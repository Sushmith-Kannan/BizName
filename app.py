import streamlit as st
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import tempfile
from PIL import Image
import io
import cv2

# Set your Hugging Face API token
HUGGING_FACE_API_KEY = "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"

# Authentication with Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_API_KEY

# Load the tokenizer and model
model_name = "facebook/opt-125m"  # Using a smaller model for efficiency
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGING_FACE_API_KEY)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HUGGING_FACE_API_KEY)

def extract_names_with_huggingface(text):
    """Extracts person and organization names using a Hugging Face model."""
    prompt = f"Extract the person name and organization name from the following text:\n\n{text}\n\nPerson Name:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    person_name = None
    organization_name = None

    lines = response_text.split('\n')
    for line in lines:
        if "Person Name:" in line:
            person_name = line.split("Person Name:")[1].strip()
        elif "Organization Name:" in line:
            organization_name = line.split("Organization Name:")[1].strip()

    return person_name, organization_name

def perform_ocr(image):
    """Perform OCR using OCR.space API."""
    api_key = "YOUR_OCR_SPACE_API_KEY"  # Replace with your OCR.space API key
    url = "https://api.ocr.space/parse/image"
    
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    
    response = requests.post(url, files={"image": img_bytes}, data={"apikey": api_key})
    result = response.json()
    
    if result["OCRExitCode"] == 1:
        return result["ParsedResults"][0]["ParsedText"]
    else:
        return "Error performing OCR."

st.title("OCR and Text Extraction App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform OCR on the image using OCR.space API
    ocr_text = perform_ocr(image)

    st.subheader("Extracted Text:")
    st.text(ocr_text)

    # Extract names using Hugging Face model
    person, organization = extract_names_with_huggingface(ocr_text)

    st.subheader("Extracted Information:")
    st.write(f"Person Name: {person}")
    st.write(f"Organization Name: {organization}")
