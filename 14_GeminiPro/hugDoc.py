# This code will extract the data from the document with Paddle ocr then it will ask the question on that extracted data.

from dotenv import load_dotenv
import os
import mimetypes
import json
import base64
from PIL import Image
from paddleocr import PaddleOCR  # Correct import
from transformers import pipeline

## Configuring Hugging Face API Key
load_dotenv()  # Ensure this line is executed to load the environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR with the right arguments

# Load the Hugging Face model using a question-answering pipeline
qa_pipeline = pipeline("question-answering", model="mrm8488/bert-small-finetuned-squadv2", use_auth_token=api_key)

def input_image_setup(file_path):
    if file_path is not None and os.path.exists(file_path):
        # Determine the MIME type based on the file extension
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type and mime_type.startswith('image/'):
            with open(file_path, "rb") as file:
                bytes_data = file.read()

            image_parts = {
                "mime_type": mime_type,
                "data": bytes_data,
            }
            return image_parts
        else:
            raise ValueError("Unsupported file type. Please provide a valid image file.")
    else:
        raise FileNotFoundError("No file provided or file does not exist.")

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path):
    # Extract text using PaddleOCR
    result = ocr.ocr(image_path)
    extracted_text = " ".join([line[1][0] for line in result[0]])  # Collect text from OCR results
    return extracted_text

def get_model_response(question, context):
    # Use the Hugging Face question-answering model
    response = qa_pipeline(question=question, context=context)
    return response["answer"]

def main():
    # Provide the path to the image file
    image_file_path = "sample_image6.jpeg"  # Change this to your image file path

    # Display the uploaded image
    image = Image.open(image_file_path)
    image.show()

    # Convert the uploaded file to bytes and process
    image_data = input_image_setup(image_file_path)

    # Extract text from the image using OCR
    extracted_text = extract_text_from_image(image_file_path)
    print(f"Extracted Text from Image: {extracted_text}")

    # Ask questions directly to the extracted text
    question = "What is the name?"  # Example question, you can ask other relevant questions
    name = get_model_response(question, extracted_text)
    
    question = "What is the date of birth?"
    dob = get_model_response(question, extracted_text)

    question = "What is the gender?"
    gender = get_model_response(question, extracted_text)

    # Constructing the final JSON response
    response_json = {
        "name": name,
        "DOB": dob,
        "gender": gender
    }

    print("Extracted Information:")
    print(json.dumps(response_json, indent=4))

if __name__ == "__main__":
    main()
