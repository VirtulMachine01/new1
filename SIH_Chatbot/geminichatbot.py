from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

## Configuring API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state variables
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'image_caption' not in st.session_state:
    st.session_state.image_caption = None
if 'image_data' not in st.session_state:
    st.session_state.image_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def generate_image_caption(image_data):
    """
    Generates a caption for the uploaded image using the Gemini model.
    """
    if st.session_state.chat is None:
        # Start a new chat
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.chat = model.start_chat()

    # Define a prompt to generate image caption
    caption_prompt = "Caption the given image"
    response = st.session_state.chat.send_message([caption_prompt, image_data[0]])
    
    # Store the caption in session state
    st.session_state.image_caption = response.text
    return st.session_state.image_caption

def get_gemini_response(user_input, prompt):
    """
    Handles user follow-up questions related to the image and chat history.
    """
    response = st.session_state.chat.send_message([user_input, st.session_state.image_data[0], prompt])
    
    # Trim chat history to the last 10 interactions
    if len(st.session_state.chat.history) > 10:
        st.session_state.chat.history = st.session_state.chat.history[-10:]
    
    # Append the response to chat history
    st.session_state.chat_history.append({
        'user': user_input,
        'bot': response.text
    })
    
    return response.text

def input_image_setup(uploaded_file):
    """
    Process the uploaded image and convert it into the required format for the model.
    """
    if uploaded_file is not None:
        # Read the file into bytes
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

# Streamlit UI
st.set_page_config(page_title="Responsive Gemini Chatbot")

st.header("Responsive Gemini Image Chatbot")

# If no image has been uploaded yet, ask for image upload
if st.session_state.image_data is None:
    st.write("Please upload an image to start the chat.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Process the uploaded image
        st.session_state.image_data = input_image_setup(uploaded_file)
        image = Image.open(uploaded_file)
        # st.image(image, caption="Uploaded Image", width=400)
        
        # Automatically generate and display the image caption
        st.session_state.image_caption = generate_image_caption(st.session_state.image_data)
        # st.subheader("Image Caption:")
        # st.write(st.session_state.image_caption)

# If an image has been uploaded, allow the user to start chatting
if st.session_state.image_data is not None:
    st.subheader("Chat about the image")

    # Display the image and previous chats
    st.image(st.session_state.image_data[0]['data'], caption="Uploaded Image", width=400)
    
    if st.session_state.image_caption:
        st.write("Caption: " + st.session_state.image_caption)
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
    # ShowChat()
    
    user_input = st.text_input("Ask a question about the image:", key="user_input")
    
    if st.button("Submit Question"):
        input_prompt = """
        You are a Conversational Chatbot which is able to understand the input image deeply.
        You have to answer the question based on the image.
        You are a chatbot so you have the memory so you have to answer the question with the follow up questions from the previous response.
        If the question is not related to the image and previous response, respond with "We will talk only about the image" only don't write another in that reponse.
        """
        # Get the chatbot response
        response = get_gemini_response(user_input, input_prompt)
        st.subheader("Answer:")
        st.write(response)

    # st.write("You can only chat about the first uploaded image. No other images can be uploaded.")






















# from dotenv import load_dotenv
# load_dotenv()

# import streamlit as st
# import os
# from PIL import Image
# import google.generativeai as genai

# ## Configuiring API Key
# genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# # Initialize chat in session state if not already done
# if 'chat' not in st.session_state:
#     st.session_state.chat = None

# def get_gemini_response(input, image, prompt):
#     if st.session_state.chat is None:
#         # Start a new chat if not already started
#         model = genai.GenerativeModel('gemini-1.5-flash')
#         st.session_state.chat = model.start_chat()

#     # Send a message and maintain the chat history
#     response = st.session_state.chat.send_message([input, image[0], prompt])

#     # # Trim chat history to last 10 messages
#     # if len(st.session_state.chat.history) > 10:
#     #     st.session_state.chat.history = st.session_state.chat.history[-10:]

#     print(st.session_state.chat.history)  # For debugging
#     return response.text

# def input_image_setup(uploaded_file):
#     if uploaded_file is not None:
#         #Read the file into bytes
#         bytes_data = uploaded_file.getvalue()

#         image_parts = [
#             {
#                 "mime_type": uploaded_file.type,
#                 "data": bytes_data,
#             }
#         ]
#         return image_parts
#     else:
#         raise FileNotFoundError("No file uploaded")
    

# ## Initialize Streamlit app
# st.set_page_config(page_title="Gemini Image Demo")
# st.header("Gemini Application")

# input = st.text_input("Input Prompt:", key="input")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# image = ""
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", width=400)

# submit = st.button("Image Information")

# input_prompt = """
# You are a Conversational Chatbot which is able to understand the input image deeply.
# You have to answer the question based on the image.
# You are a chatbot so you have the memory so you have to answer the question with the follow up questions from the previous response.
# If the question is not related to the image and previous response, respond with "We will talk only about the image" only don't write another in that reponse.
# """

# if submit:
#     image_data = input_image_setup(uploaded_file)
#     response = get_gemini_response(input, image_data, input_prompt)
#     st.subheader("Answer:")
#     st.write(response)


# from deep_translator import GoogleTranslator
# def translate_text(text, dest_language='en'):
#     translator = GoogleTranslator(source='auto', target=dest_language)
#     translation = translator.translate(text)
#     return translation