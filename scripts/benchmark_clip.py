import torch
import time
import openvino as ov
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path

# Config
model_id = "openai/clip-vit-large-patch14"
ov_model_path = r"F:\Porting_Project\models_ir\clip_vit_l14\clip_vit_l14.xml"
text_prompt = "A cinematic Vedic deity in a golden temple, high detail, 8k"

# 1. Benchmark PyTorch (Baseline)
print("--- Benchmarking PyTorch (Baseline) ---")
tokenizer = CLIPTokenizer.from_pretrained(model_id)
pt_model = CLIPTextModel.from_pretrained(model_id).to("cpu") # Testing on CPU first
inputs = tokenizer(text_prompt, return_tensors="pt")

start = time.perf_counter()
for _ in range(50):
    with torch.no_grad():
        pt_model(**inputs)
pt_time = (time.perf_counter() - start) / 50
print(f"PyTorch Avg Latency: {pt_time:.4f} seconds")

# 2. Benchmark OpenVINO
print("\n--- Benchmarking OpenVINO (Optimized) ---")
core = ov.Core()
ov_model = core.read_model(ov_model_path)
compiled_model = core.compile_model(ov_model, "CPU") # Target i5-11400

ov_inputs = { "input_ids": inputs["input_ids"].numpy() }

start = time.perf_counter()
for _ in range(50):
    compiled_model(ov_inputs)
ov_time = (time.perf_counter() - start) / 50
print(f"OpenVINO Avg Latency: {ov_time:.4f} seconds")

print(f"\n--- RESULTS ---")
print(f"Speedup: {pt_time / ov_time:.2f}x")