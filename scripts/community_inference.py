import numpy as np
import openvino as ov
from transformers import CLIPTokenizer
from PIL import Image
import os

# Setup Paths
ov_clip_path = "F:/Porting_Project/models_ir/clip_vit_l14/clip_vit_l14.xml"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# 1. Generate the "Thought" (CLIP Inference on Intel CPU)
print("--- Step 1: CLIP Text Encoding (Intel OpenVINO) ---")
prompt = "A cinematic Vedic temple in a lush jungle, high quality, 4k"
inputs = tokenizer(prompt, padding='max_length', max_length=77, return_tensors='np')
core = ov.Core()
model = core.read_model(ov_clip_path)
compiled_model = core.compile_model(model, "CPU")

# Model expects 'input_ids' based on our previous diagnosis
output = compiled_model({"input_ids": inputs['input_ids']})
last_hidden_state = output[0]
print(f"Encoded Prompt Shape: {last_hidden_state.shape}")

# 2. Simulate the 'Visual Muscle' (VAE Prep)
print("\n--- Step 2: VAE Decoding Preparation ---")
# In a full pipeline, the UNet would go here. For now, we simulate a latent.
# We'll create a random latent to prove the VAE can decode it into an image.
latent = np.random.randn(1, 4, 64, 64).astype(np.float32)
latent.tofile("F:/Porting_Project/scripts/test_latent.raw")

print(f"Latent created and saved for VAE. Ready for Qualcomm NPU verification.")
print("\n--- Next Steps for the Masses ---")
print("Run the following to decode this latent using your new INT8 VAE:")
print("qnn-net-run --model F:/Porting_Project/models_dlc/vae_decoder_int8.dlc --input_list F:/Porting_Project/vae_input_list.txt --backend libQnnCpu.dll")