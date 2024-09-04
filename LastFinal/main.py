from flask import Flask, render_template, request, jsonify, redirect, url_for
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import numpy as np
import torch
import json
import spacy
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator
from urllib.parse import quote, unquote

isGPU = True
if(isGPU):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Change to 'en' if needed

# Generate tokens from the caption
def GenerateTokens_FromCaption(generatedCaption):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(generatedCaption)
    objects = []
    for chunk in doc.noun_chunks:
        phrase = ' '.join(token.text for token in chunk if token.text.lower() not in ('a', 'an', 'the', 'picture'))
        if phrase.strip():
            objects.append(phrase.strip())

    unique_objects = list(set(objects))
    return unique_objects

# Extract text from the image
def extract_text(image):
    # Convert PIL image to numpy array
    img_array = np.array(image)
    # Use PaddleOCR to perform OCR
    results = ocr.ocr(img_array, cls=True)
    # extracted_text = ""
    # for result in results:
    #     for line in result:
    #         extracted_text += line[1][0] + " "  # Append recognized text
    # return extracted_text.strip()
    texts_with_positions = []
    for result in results:
        for line in result:
            text = line[1][0]
            texts_with_positions.append(text)
    return texts_with_positions


def translate_text(text, dest_language='en'):
    translated = GoogleTranslator(source='zh-CN', target=dest_language).translate(text)
    return translated

app = Flask(__name__)

# Load model and processor
processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Process the uploaded image
        print("Processing image...")
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        print("Image opened successfully.")
        text = "a picture of "
        inputs = processor(img, text, return_tensors="pt").to(device) 

        print("check-2")
        # Generate caption
        out = model.generate(**inputs, num_beams=3).to(device)
        generated_text = processor.decode(out[0], skip_special_tokens=True)

        print("check-1")
        tokensFromCaption = GenerateTokens_FromCaption(generated_text)
        ocr_text = extract_text(img)  # Pass the PIL image directly
        translated_text = [translate_text(text) for text in ocr_text]

        # Create a new entry for the uploaded image
        new_entry = {
            "image_id": file.filename,
            "image_caption_tokens": tokensFromCaption,
            "image_ocr_text": translated_text
        }
        print("check1")
        return jsonify(new_entry)

    except Exception as e:
        print("IsError")
        print(f"Error: {e}")  # Log the error
        return jsonify({"error": str(e)}), 500  # Return the error message

@app.route('/result')
def result():
    # Retrieve the JSON data from the request
    data = request.args.get('data')
    if data:
        # Parse the JSON data
        data_dict = json.loads(data)
        # Render the result.html template with the data
        return render_template('result.html', data = data_dict)
    else:
        # Handle the case when no data is provided
        return render_template('result.html', data={})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)