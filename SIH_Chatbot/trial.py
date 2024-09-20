from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

# Load Qwen2-VL model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="cpu")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Initialize memory
memory = ConversationBufferWindowMemory(window_size=10)

# Initialize ConversationChain
conversation_chain = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True
)

def generate_response(image_path=None, user_text=""):
    # Prepare user input and image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": user_text
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Process inputs for the model
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Generate the response
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Store the response in memory
    conversation_chain.memory.save_context({"input": user_text}, {"output": output_text[0]})
    
    return output_text[0]

def generate_follow_up_response(user_text):
    # Add the follow-up question to the conversation
    follow_up_response = conversation_chain({"input": user_text})
    
    # Store the follow-up response in memory
    conversation_chain.memory.save_context({"input": user_text}, {"output": follow_up_response["output"]})
    
    return follow_up_response["output"]

# Example usage
initial_response = generate_response(image_path="hotel.jpeg", user_text="Describe the image")
print("Initial Response:", initial_response)

# Example usage for follow-up questions
follow_up_response = generate_follow_up_response("count the total words in the last description.")
print("Follow-Up Response:", follow_up_response)
