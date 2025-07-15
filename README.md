# Image Super-Resolution and Object Detection

This repository contains code and experiments from a master's thesis project titled:

**"Enhancing Sperm Detection in Microscopic Videos Using Image Super-Resolution: An Experimental Study on Detection Accuracy and Generalization"**

The study explores how classical and deep learning-based super-resolution techniques (e.g., Swin2SR and Real-ESRGAN) impact the performance of YOLOv11-based object detection on human sperm microscopy frames.

---

## Experimental Pipelines

The following enhancement pipelines were implemented and evaluated:

1. **Baseline** – Native resolution YOLOv11 training.
2. **Classical Interpolation** – Downscaling to lower resolutions then upscaling via bicubic interpolation.
3. **Swin2SR** – Super-resolution using Swin Transformer.
4. **Real-ESRGAN** – Real-ESRGAN-based upscaling.
5. **Native Resolution SR** – Applying SR on original native-resolution data.

---

## Project Structure

```text
Image_super_resolution_and_object_detection/
├── scripts/             
│   ├── baseline/
│   ├── interpolation/
│   ├── swin2sr/
│   ├── realesrgan/
│   └── native_sr/
├── utils/               
├── results/             
├── models/              
├── datasets/            
├── README.md
├── requirements.txt
└── LICENSE

