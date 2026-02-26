import numpy as np
import os

# Create folder for VAE calibration data
calib_dir = "F:/Porting_Project/calib_data/vae"
os.makedirs(calib_dir, exist_ok=True)

# Generate 10 sample latent files
for i in range(10):
    sample = np.random.randn(1, 4, 64, 64).astype(np.float32)
    sample.tofile(f"{calib_dir}/latent_{i}.raw")

# Create the input list file
with open("F:/Porting_Project/vae_input_list.txt", "w") as f:
    for i in range(10):
        f.write(f"latent_sample:={calib_dir}/latent_{i}.raw\n")

print("VAE Calibration data and input_list generated!")