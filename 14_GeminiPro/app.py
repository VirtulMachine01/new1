## Invoice extractor

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

## Configuiring API Key
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

#Function to load Gemini Pro vision model and get response
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content([input, image[0], prompt])
        
        # Check if the response was blocked due to safety ratings
        if response.candidates and response.candidates[0].content:
            return response.candidates[0].content.parts[0].text
        else:
            # Check safety ratings
            if response.candidates and response.candidates[0].safety_ratings:
                safety_issues = [f"{rating.category}: {rating.probability}"
                                 for rating in response.candidates[0].safety_ratings
                                 if rating.probability != "NEGLIGIBLE"]
                return f"Response blocked due to safety concerns: {', '.join(safety_issues)}"
            else:
                return "No valid response generated. Please try again with a different prompt or image."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# def get_gemini_response(input, image, prompt):
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content([input, image[0], prompt])
#     return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        #Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    

## initialize out streamlit app

st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemini Application")
input=st.text_input("Input Prompt: ", key="input")
# input = """
# What is name?
# What is DOB?
# What is Gender?
# What is Number?
# Whch Document is this?
# """

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
image = ""
if uploaded_file is not None:
    image  = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me about the Invoice")

input_prompt="""
You are an expert in understanding Indian Documents like Aadhaar card and Pancard.
You will recieve input document as image.
and you will have to answer questions based on the input image.
"""
# input_prompt="""
# You are an expert in understanding invoices. You will recieve input images as invoices 
# and you will have to answer questions based on the input image.
# """

if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("Answer : ")
    st.write(response)

# if submit:
#     image_data = input_image_setup(uploaded_file)
#     response = get_gemini_response(input_prompt, image_data, input)
    
#     st.subheader("Response:")
    
#     if response is None:
#         st.error("No valid response generated. Please give a proper prompt.")
#     elif response.startswith("Response blocked due to safety concerns:"):
#         st.warning("Change the prompt")
#     elif response.startswith("An error occurred:"):
#         st.error("Change the prompt")
#     else:
#         st.write(response)

#Improve the model

# Try multishot prompting
# finetune the model
# Try with OCR