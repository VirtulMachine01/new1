# from dotenv import load_dotenv
# load_dotenv()
# import os
# import mimetypes
# import json
# from PIL import Image
# import google.generativeai as genai

# ## Configuiring API Key
# genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# ##Function to load Gemini Pro vision model and get response
# def get_gemini_response(input, image, prompt):
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     try:
#         response = model.generate_content([input, image[0], prompt])
        
#         # Check if the response was blocked due to safety ratings
#         if response.candidates and response.candidates[0].content:
#             return response.candidates[0].content.parts[0].text
#         else:
#             # Check safety ratings
#             if response.candidates and response.candidates[0].safety_ratings:
#                 safety_issues = [f"{rating.category}: {rating.probability}"
#                                  for rating in response.candidates[0].safety_ratings
#                                  if rating.probability != "NEGLIGIBLE"]
#                 return f"Response blocked due to safety concerns: {', '.join(safety_issues)}"
#             else:
#                 return "No valid response generated. Please try again with a different prompt or image."
#     except Exception as e:
#         return f"An error occurred: {str(e)}"

# def input_image_setup(file_path):
#     if file_path is not None and os.path.exists(file_path):
#         # Determine the MIME type based on the file extension
#         mime_type, _ = mimetypes.guess_type(file_path)
        
#         if mime_type and mime_type.startswith('image/'):
#             with open(file_path, "rb") as file:
#                 bytes_data = file.read()

#             image_parts = [
#                 {
#                     "mime_type": mime_type,
#                     "data": bytes_data,
#                 }
#             ]
#             return image_parts
#         else:
#             raise ValueError("Unsupported file type. Please provide a valid image file.")
#     else:
#         raise FileNotFoundError("No file provided or file does not exist.")
    
# input = """
# Extract the name, DOB, gender JSON format from the provided document.
# """
# uploaded_file = "pan.jpeg"
# image  = Image.open(uploaded_file)

# input_prompt="""
# You are an expert in understanding Indian Documents like Aadhaar card and Pancard.
# You will recieve input document as image.
# and you will have to answer questions based on the input image and provide the response in JSON format.
# Dont write json and ``` in the output.only return the json.
# """
# image_data = input_image_setup(uploaded_file)
# response = get_gemini_response(input_prompt, image_data, input)

# if response is None:
#     print("No valid response generated. Please give a proper prompt.")
# elif response.startswith("Response blocked due to safety concerns:"):
#     print("Change the prompt")
# elif response.startswith("An error occurred:"):
#     print("Change the prompt")
# else:
#     try:
#         response_json = json.loads(response)
#         print(json.dumps(response_json, indent=4))
#     except json.JSONDecodeError:
#         print("The response is not a valid JSON. Here is the response text:")
#         print(response)











from dotenv import load_dotenv
load_dotenv()
import os
import mimetypes
import json
import base64
from PIL import Image
import google.generativeai as genai

## Configuring API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Gemini Pro vision model and get response
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

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode example images
example_image1_base64 = encode_image_to_base64("sample_image1.jpg")
example_image2_base64 = encode_image_to_base64("sample_image2.png")
example_image3_base64 = encode_image_to_base64("sample_image3.png")
example_image4_base64 = encode_image_to_base64("sample_image4.png")
# example_image5_base64 = encode_image_to_base64("sample_image5.jpeg")

# Few-shot examples
few_shot_examples = f"""
Example 1:
Input: Extract the name, DOB, gender, Aadhar Number from the provided Aadharcard Document.
Image (base64): {example_image1_base64}
Output: {{ "name": "Abhishek Tiwari", "DOB": "07/05/2002", "gender": "Male", "Aadhar_Number": "817329973675"}}
"""
# Example 2:
# Input: Extract the name, DOB, gender, Aadhar Number from the provided Aadharcard Document.
# Image (base64): {example_image2_base64}
# Output: {{ "name": "Jairam Suryalal Yadav", "DOB": "31/12/1988", "gender": "Male", "Aadhar_Number": "855407974578"}}

# Example 3:
# Input: Extract the name, DOB, gender, Aadhar Number from the provided Aadharcard Document.
# Image (base64): {example_image3_base64}
# Output: {{ "name": "Adarsh kumar", "DOB": "17/06/1995", "gender": "Male", "Aadhar_Number": "846550732129"}}

# Example 4:
# Input: Extract the name, DOB, gender, Aadhar Number from the provided Aadharcard Document.
# Image (base64): {example_image4_base64}
# Output: {{ "name": "Sakhi bai kushwah", "DOB": "10/10/1989", "gender": "Female", "Aadhar_Number": "982663598852"}}

# Example 5:
# Input: Extract the name, DOB, gender, Aadhar Number from the provided Aadharcard Document.
# Image (base64): {example_image5_base64}
# Output: {{ "name": "Santosh kumar", "DOB": "13/07/1982", "gender": "Male", "Aadhar_Number": "631723766121"}}
# """

input_text = """
Extract the name, DOB, gender, Aadhar Number from the provided Aadharcard Document.
"""

uploaded_file = "sample_image4.png"
image  = Image.open(uploaded_file)

input_prompt = f"""
You are an expert in understanding Indian Documents like Aadhaar card.
You will receive input document as image.
You will have to answer questions based on the input image and provide the response in JSON format.
Don't write json and ``` in the output. Only return the json.

{few_shot_examples}

Now, given the document, {input_text}
"""

image_data = input_image_setup(uploaded_file)
response = get_gemini_response(input_prompt, image_data, input_text)

if response is None:
    print("No valid response generated. Please give a proper prompt.")
elif response.startswith("Response blocked due to safety concerns:"):
    print("Change the prompt")
elif response.startswith("An error occurred:"):
    print("Change the prompt")
else:
    try:
        response_json = json.loads(response)
        print(json.dumps(response_json, indent=4))
    except json.JSONDecodeError:
        print("The response is not a valid JSON. Here is the response text:")
        print(response)









#Improve the model

# Try multishot prompting with document type
# finetune the model
# Try with OCR