import torch
from transformers import CLIPTextModel, CLIPTokenizer
import openvino as ov
from pathlib import Path

# Setup paths
save_dir = Path(r"F:\Porting_Project\models_ir\clip_vit_l14")
save_dir.mkdir(parents=True, exist_ok=True)

model_id = "openai/clip-vit-large-patch14"
print(f"Loading {model_id} from Hugging Face...")

# Load Tokenizer and Model
tokenizer = CLIPTokenizer.from_pretrained(model_id)
model = CLIPTextModel.from_pretrained(model_id)

# Prepare dummy input for conversion (CLIP uses 77 tokens)
dummy_input = torch.ones([1, 77], dtype=torch.long)

print("Converting to OpenVINO Intermediate Representation (IR)...")
# Convert the PyTorch model directly to OpenVINO format
ov_model = ov.convert_model(model, example_input=dummy_input)

# Save as FP16 to save space on your F: drive and speed up inference
ov.save_model(ov_model, save_dir / "clip_vit_l14.xml", compress_to_fp16=True)

print(f"\n--- SUCCESS ---")
print(f"Model saved at: {save_dir}")
print(f"Files created: clip_vit_l14.xml, clip_vit_l14.bin")