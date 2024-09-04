import streamlit as st
from PIL import Image
import numpy as np
import torch
import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator

isGPU = True

if(isGPU):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

#model for captioning
processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)

# model for OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Function to generate tokens from caption
def generate_tokens_from_caption(generated_caption):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(generated_caption)
    objects = []
    for chunk in doc.noun_chunks:
        phrase = ' '.join(token.text for token in chunk if token.text.lower() not in ('a', 'an', 'the', 'picture'))
        if phrase.strip():
            objects.append(phrase.strip())
    return list(set(objects))

# Function to extract text from the image
def extract_text(image):
    img_array = np.array(image)
    results = ocr.ocr(img_array, cls=True)
    texts_with_positions = [line[1][0] for result in results for line in result]
    return texts_with_positions

# Function to translate text
def translate_text(text, dest_language='en'):
    return GoogleTranslator(source='auto', target=dest_language).translate(text)


# Initial webpage for dictionary input
def main():
    # Streamlit application
    st.title("Image Captioning and OCR")

    # Image upload
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    print("check0")
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        # st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image
        if st.button("Get Data"):
            with st.spinner("Collecting Data From Image..."):
                # Prepare input for the model
                text = "a picture of "
                inputs = processor(image, text, return_tensors="pt").to(device)

                # Generate caption
                out = model.generate(**inputs, num_beams=3)
                generated_text = processor.decode(out[0], skip_special_tokens=True)

                # Generate tokens and extract text
                tokens_from_caption = generate_tokens_from_caption(generated_text)
                ocr_text = extract_text(image)
                translated_text = [translate_text(text) for text in ocr_text]

                # Create a JSON object
                result_json = {
                    "image_id": uploaded_file.name,
                    "image_caption_tokens": tokens_from_caption,
                    "image_ocr_text": translated_text
                }

                st.session_state.json_object = result_json
                st.rerun()

    
# Results page to display JSON object
def results_page():
    # st.title("JSON Object Result")

    if 'json_object' in st.session_state:
        json_object = st.session_state.json_object
        st.json(json_object)  # Display JSON object
    else:
        st.error("No JSON object found. Please go back and enter a dictionary.")
    
    # # Optionally, you can provide a download button for the JSON
    # json_string = json.dumps(result_json, indent=2)
    # st.download_button("Download JSON", json_string, file_name="result.json", mime="application/json")

if __name__ == "__main__":
    if 'json_object' not in st.session_state:
        main()
    else:
        results_page()