# 🌌 Celeste-Bridge: Inclusive AI for the Masses
> **"Every Penny, Watt, and VRAM Count."**

Celeste-Bridge is an infrastructure project dedicated to breaking the "Hardware Wall." We specialize in **Hybrid Inference** and **Model Porting**, ensuring that high-fidelity AI models—like SDXL—run smoothly on everyday hardware: CPUs, Snapdragon NPUs, and Low-VRAM GPUs.

---

## 🚀 The Mission: Filling the Gap
High-end AI shouldn't require a $2,000 GPU. Our goal is to distribute the system load evenly across all available silicon:
* **CPU Optimization**: Leveraging Intel OpenVINO for high-speed VAE/CLIP decoding.
* **NPU Acceleration**: Native support for Qualcomm Snapdragon (QNN) to offload background tasks.
* **Hybrid Orchestration**: Running the heavy UNet on GPU while keeping the rest on the CPU to save ~4.2GB of VRAM.

---

## 🛠️ The Technical Stack
This repository contains the "Blueprints" for our efficiency pipeline:
* **`workflows/SDXL I2I.json`**: Our "Data Factory" for consistent character generation (Testing: Ayesha).
* **`scripts/port_vae.py`**: Automated conversion to OpenVINO IR format.
* **`metadata/`**: Live efficiency logs proving VRAM savings and power reduction.

---

## 📊 Performance Proof (Hybrid Mode)
Based on our latest `hybrid_efficiency_report.json`:
| Component | Device | VRAM Impact | Power Draw |
| :--- | :--- | :--- | :--- |
| **UNet** | GPU (RTX) | 6.0 GB | High |
| **VAE** | **CPU (OpenVINO)** | **0.0 GB** | **Low** |
| **CLIP** | **CPU (OpenVINO)** | **0.0 GB** | **Low** |
| **Total Saved** | | **~4.2 GB** | **Optimized** |

---

## ⚙️ Setup for Developers
1. **Clone the Blueprints**: `git clone https://github.com/Celeste-Imperia/Celeste-Bridge.git`
2. **Set your Credentials**: 
   ```powershell
   $env:HF_TOKEN='your_token_here'
   ```
3. **Active Development Workflow**: We are currently building the conversion pipeline to move models from high-VRAM environments to a **Masses-Ready** stack.
   * **Data Factory**: Utilizing `workflows/SDXL I2I.json` to generate consistent character test sets (e.g., Ayesha).
   * **Conversion Logic**: Baseline scripts for OpenVINO IR conversion and NPU targeting.
   * **Target Hardware**: Testing optimizations for Intel CPUs and Qualcomm Snapdragon NPUs.

---

## 🌟 The Forge & The Mission
This project uses an **NVIDIA RTX A4000 (16GB)** as a high-capacity development forge. 
* **The Goal**: We use the A4000's headroom to train, calibrate, and port high-fidelity models.
* **The Result**: We output "inclusive weights" that run on standard CPUs and Snapdragon NPUs, filling the gap for users without dedicated graphics hardware.

---

## 🤝 Acknowledgments
This project is built for the community. We use the RTX A4000 as a development forge to create tools that work for users with no GPU at all.
FINAL_README_CELESTE_BRIDGE.txt
Displaying FINAL_README_CELESTE_BRIDGE.txt.

