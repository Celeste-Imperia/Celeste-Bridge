import torch
from diffusers import AutoencoderTiny
import os

# Setup paths
save_dir = "F:/Porting_Project/models_ir/vae_decoder"
os.makedirs(save_dir, exist_ok=True)

print("Loading TinyVAE (TAESD) and materializing weights on CPU...")
# Correct class name is AutoencoderTiny
vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
decoder = vae.decoder.to("cpu")
decoder.eval()

# Create dummy input strictly on CPU
dummy_input = torch.randn(1, 4, 64, 64).to("cpu")

print("Exporting to ONNX using Stable Legacy Path...")
with torch.no_grad():
    torch.onnx.export(
        decoder, 
        dummy_input, 
        f"{save_dir}/vae_decoder.onnx", 
        input_names=['latent_sample'], 
        output_names=['pixel_values'],
        opset_version=14, # Qualcomm-friendly opset
        do_constant_folding=True,
        export_params=True
    )

print(f"\nSuccess! TinyVAE Decoder exported to: {save_dir}/vae_decoder.onnx")