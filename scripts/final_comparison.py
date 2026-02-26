import time
import numpy as np
import openvino as ov
from transformers import CLIPTokenizer

# Setup paths
ov_model_path = "F:/Porting_Project/models_ir/clip_vit_l14/clip_vit_l14.xml"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
prompt = "A high-quality cinematic shot for a business presentation"

# 1. OpenVINO Inference (Intel CPU)
core = ov.Core()
ov_model = core.read_model(ov_model_path)
compiled_ov = core.compile_model(ov_model, "CPU")

# The model only expects 'input_ids'
inputs = tokenizer(prompt, padding='max_length', max_length=77, return_tensors='np')
ov_inputs = {
    "input_ids": inputs['input_ids']
}

print("\n--- Starting Intel OpenVINO Benchmark ---")
# Warm up
compiled_ov(ov_inputs)

start = time.perf_counter()
for _ in range(10):
    compiled_ov(ov_inputs)
end = time.perf_counter()
print(f"Average OpenVINO Latency: {(end - start) / 10 * 1000:.2f} ms")

# 2. Qualcomm NPU Instruction
print("\n--- Qualcomm NPU Instructions ---")
print("To benchmark the Qualcomm port, run the following in your terminal:")
print("qnn-net-run --model F:/Porting_Project/models_dlc/clip_vit_l14_int8.dlc --input_list F:/Porting_Project/input_list.txt --backend libQnnCpu.dll")