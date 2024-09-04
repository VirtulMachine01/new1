from flask import Flask, render_template
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import json
import os
import spacy
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator

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

#Extract text from the image
def extract_text(image_path):
    # Create a PaddleOCR instance
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # Use Chinese language
    # Perform OCR
    results = ocr.ocr(image_path, cls=True)
    
    # Extract and return text
    extracted_text = ""
    for result in results:
        for line in result:
            extracted_text += line[1][0] + " "  # Append recognized text
    return extracted_text.strip()

def translate_text(text, dest_language='en'):
    translated = GoogleTranslator(source='zh-CN', target=dest_language).translate(text)
    return translated



app = Flask(__name__)

# Load model and processor
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap")

# Specify the directory containing images
IMAGE_DIR = 'static/'
RESULTS_FILE = 'results.json'

# Load existing results if the file exists
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        existing_results = json.load(f)
else:
    existing_results = []



@app.route('/')
def index():
    # Get a list of image files in the static directory
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Initialize a list to hold captions
    captions = []

    # Process each image and generate captions
    for image_file in image_files:  # Process all images

        if any(result['image_id'] == image_file for result in existing_results):
            # Skip processing if already exists
            continue

        img_path = os.path.join(IMAGE_DIR, image_file)
        img = Image.open(img_path).convert('RGB')
        text = "a picture of "
        inputs = processor(img, text, return_tensors="pt")

        # Generate caption
        out = model.generate(**inputs, num_beams=3)
        generated_text = processor.decode(out[0], skip_special_tokens=True)

        tokensFromCaption = GenerateTokens_FromCaption(generated_text)

        ocr_text = extract_text(img_path)
        translated_text = translate_text(ocr_text, 'en')

        # Append a dictionary with image_id and image_caption to the list
        new_entry = {
            "image_id": image_file,
            "image_caption_tokens": tokensFromCaption,
            "image_ocr_text": translated_text
        }

        captions.append(new_entry)
        # Update existing results
        existing_results.append(new_entry)
        
    # Save updated results to JSON file
    with open(RESULTS_FILE, 'w') as f:
        # json.dump(existing_results, f, indent=4)
        json.dump(existing_results, f, separators=(',', ':'))

    # Convert the captions list to JSON
    # captions_json = json.dumps(existing_results, indent=4)
    captions_json = json.dumps(existing_results, separators=(',', ':'))

    # Render the captions on the main page
    return render_template('index.html', captions=captions_json)
    # return Response(captions_json, mimetype='application/json') # Remaining for check

if __name__ == '__main__':
    # app.run(debug=True) #localhost only
    app.run(host='0.0.0.0', port=5000, debug=True)