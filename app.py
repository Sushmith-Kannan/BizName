import streamlit as st
from paddleocr import PaddleOCR
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import tempfile
from PIL import Image
import io

# Set your Hugging Face API token
HUGGING_FACE_API_KEY = "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"

# Authentication with Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_API_KEY

# Load the tokenizer and model
model_name = "facebook/opt-125m"  # Using a smaller model for efficiency
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGING_FACE_API_KEY)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HUGGING_FACE_API_KEY)

ocr_model = PaddleOCR(lang='en')

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

st.title("OCR and Text Extraction App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_path = tempfile.mktemp(suffix=".jpeg")

    with open(img_path, "wb") as f:
        f.write(img_bytes.read())

    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Perform OCR on the image
    result = ocr_model.ocr(img_path)
    
    # Extract text from OCR result
    ocr_text = ""
    for line in result:
        ocr_text += ' '.join([word_info[1][0] for word_info in line]) + '\n'
    
    st.subheader("Extracted Text:")
    st.text(ocr_text)

    # Extract names using Hugging Face model
    person, organization = extract_names_with_huggingface(ocr_text)

    st.subheader("Extracted Information:")
    st.write(f"Person Name: {person}")
    st.write(f"Organization Name: {organization}")

    # Clean up temporary files
    os.remove(img_path)
