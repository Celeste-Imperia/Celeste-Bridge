import numpy as np
import os
from transformers import CLIPTokenizer

# 1. Setup paths
data_dir = "F:/Porting_Project/calibration_data"
os.makedirs(data_dir, exist_ok=True)

# 2. Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
prompts = [
    "a professional office building with glass windows",
    "a high-resolution photograph of a futuristic city",
    "a cozy living room with a fireplace and a bookshelf",
    "a detailed map of a fantasy world",
    "a clean, modern logo for a tech startup"
]

# 3. Save raw binary inputs
with open("F:/Porting_Project/input_list.txt", "w") as f:
    for i, p in enumerate(prompts):
        inputs = tokenizer(p, padding='max_length', max_length=77, truncation=True, return_tensors='np')
        
        ids_path = f"{data_dir}/ids_{i}.raw"
        mask_path = f"{data_dir}/mask_{i}.raw"
        
        inputs['input_ids'].astype(np.int32).tofile(ids_path)
        inputs['attention_mask'].astype(np.int32).tofile(mask_path)
        
        f.write(f"input_ids:={ids_path} attention_mask:={mask_path}\n")

print("Success! Calibration data and input_list.txt are ready.")