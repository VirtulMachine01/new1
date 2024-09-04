# image captioning with streamlit as interactive application

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch

# Load model and processor
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap")

# Streamlit app title
st.title("Image Captioning for Reserch")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Prepare the image for the model
    raw_image = img.convert('RGB')
    text = "a picture of "
    inputs = processor(raw_image, text, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs, num_beams=3)
    generated_text = processor.decode(out[0], skip_special_tokens=True)

    # Display the generated caption
    st.write("Generated Caption: ", generated_text)

    # Display as a json object
    st.json({"caption": generated_text})





# web service using html and falsk
# from flask import Flask, request, jsonify, render_template
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch
# import io

# app = Flask("Image Captioning Tool")

# # Load model and processor
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
# model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/caption', methods=['POST'])
# def caption_image():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     img = Image.open(io.BytesIO(file.read())).convert('RGB')
#     text = "a picture of "
#     inputs = processor(img, text, return_tensors="pt")

#     out = model.generate(**inputs, num_beams=3)
#     generated_text = processor.decode(out[0], skip_special_tokens=True)
    
#     return jsonify({'caption': generated_text})

# if __name__ == '__main__':
#     app.run(debug=True)