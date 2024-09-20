from dotenv import load_dotenv
load_dotenv()

import os
import mimetypes
from PIL import Image
import google.generativeai as genai

## Configuring API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
chat_history = []

def input_image_setup(file_path):
    if file_path is not None and os.path.exists(file_path):
        # Determine the MIME type based on the file extension
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type and mime_type.startswith('image/'):
            with open(file_path, "rb") as file:
                bytes_data = file.read()

            image_parts = [
                {
                    "mime_type": mime_type,
                    "data": bytes_data,
                }
            ]
            return image_parts
        else:
            raise ValueError("Unsupported file type. Please provide a valid image file.")
    else:
        raise FileNotFoundError("No file provided or file does not exist.")
    
def generate_image_caption(image_data):
    chat = model.start_chat()

    # Define a prompt to generate image caption
    caption_prompt = "Caption the given image"
    prompt = "You are an Expert in the image understanding. You will get the image to you have to caption that image in formal way."
    response = chat.send_message([caption_prompt, image_data[0], prompt])
    
    # Store the caption in session state
    image_caption = response.text
    return image_caption, chat


def get_gemini_response(user_input, prompt, chat, image_data):
    response = chat.send_message([user_input, image_data[0], prompt])
    
    # Update chat history
    chat_history.append({
        'user': user_input,
        'bot': response.text
    })

    # Trim chat history to the last 10 interactions
    if len(chat.history) > 10:
        chat.history = chat.history[-10:]
    
    return response.text

image_path = "hotel.jpeg"
image_data = input_image_setup(image_path)
image_caption, chat = generate_image_caption(image_data)
print(image_caption)

prompt = """
You are a Conversational Chatbot which is able to understand the input image deeply.
You have to answer the question based on the image.
You are a chatbot so you have the memory so you have to answer the question with the follow up questions from the previous response.
If the question is not related on the provided image then explain them response will not be better and ask about the image 
"""
# If the question is not related to the image and previous response, respond with "We will talk only about the image" only don't write another in that reponse.
# """

while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break
    response = get_gemini_response(user_input, prompt, chat, image_data)
    print("Gemini: ", response)

# def print_chat_history():
#     for i, entry in enumerate(chat_history):
#         print(f"Interaction {i + 1}:")
#         print(f"User: {entry['user']}")
#         print(f"Bot: {entry['bot']}")
#         print("-" * 40)

# # Call this function where you want to print the chat history
# print_chat_history()