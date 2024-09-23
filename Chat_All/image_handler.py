from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64

def convert_bytes_to_base64(image_bytes):
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return "data:image/jpeg;base64,"+encoded_string

def handle_image(image_bytes, user_message):
    chat_handler = Llava15ChatHandler(clip_model_path="models/llava/mmproj-model-f16.gguf")
    llm = Llama(
        model_path="models/llava/ggml-model-q5_k.gguf",
        chat_handler=chat_handler,
        logits_all=True,
        n_ctx=1024 # increased to accomodate the image embedding
    )
    image_base64 = convert_bytes_to_base64(image_bytes)
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an assistant who perfectly describes images."},
            {
                "role": "user",
                "content":[
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": {"url": user_message}}
                ]
            }
        ]
    )
    print(output)
    return output["choices"][0]["message"]["content"]

def convert_img_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return "data:image/jpeg;base64,"+encoded_string
    
def generate_Img_txt_base64(image_path):
    img = convert_img_to_base64(image_path=image_path)
    with open("icons/txt/image2.txt", "w")as f:
        f.write(img)

# if __name__ == "__main__":
    # generate_Img_txt_base64("icons/user_image.png")
    # generate_Img_txt_base64("icons/bot_image.png")