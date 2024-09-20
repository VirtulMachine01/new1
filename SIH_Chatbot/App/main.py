from flask import Flask, request, render_template
from dotenv import load_dotenv
import os
import mimetypes
from werkzeug.utils import secure_filename
import google.generativeai as genai

app = Flask(__name__)
load_dotenv()

# Set up a folder to store uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to delete an image file after processing
def delete_image(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

## Configuring API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
chat_history = []

def input_image_setup(file_path):
    # if file_path is not None and os.path.exists(file_path):
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
    # else:
    #     raise FileNotFoundError("No file provided or file does not exist.")
    
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



@app.route('/', methods=['GET', 'POST'])
def index():
    image_caption=""
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Secure the filename and save the image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file to the uploads folder
            file.save(file_path)

            image_data = input_image_setup(file_path)
            image_caption, chat = generate_image_caption(image_data)
    
    return render_template('index.html', text=image_caption)

if __name__ == '__main__':
    app.run(debug=True)
