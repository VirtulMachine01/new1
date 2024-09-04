from datasets import Dataset, Features, Value, Image
import os
from PIL import Image as PILImage
import io

# Load JSON metadata
import json
with open('data/metadata.json', 'r') as f:
    metadata = json.load(f)

# Directory where images are stored
image_dir = 'data'  # Replace with your image directory path

# Prepare data
data = {
    'image': [],
    'caption': []
}

for item in metadata:
    image_path = os.path.join(image_dir, item['file_name'])
    
    # Load image
    with open(image_path, 'rb') as img_file:
        img = PILImage.open(io.BytesIO(img_file.read()))
        data['image'].append(img)
    
    # Add corresponding text
    data['caption'].append(item['caption'])

# Define the dataset features
features = Features({
    'image': Image(),
    'caption': Value('string'),
})

# Create a Dataset object
dataset = Dataset.from_dict(data, features=features)

# Push the dataset to Hugging Face
dataset.push_to_hub("VirtualMachine01/TrialDataset", split="train")
