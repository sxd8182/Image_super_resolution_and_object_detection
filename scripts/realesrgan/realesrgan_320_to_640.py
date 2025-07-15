import os
import cv2
import shutil
import pandas as pd
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
from time import time

# Real-ESRGAN dependencies
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Configs
SOURCE_TRAIN_TXT = "/mnt/users/sdevkota/train_corrected.txt"
SOURCE_VAL_TXT = "/mnt/users/sdevkota/val_corrected.txt"
SOURCE_TEST_TXT = "/mnt/users/sdevkota/test_yolo_m.txt"
MODEL_AR = "yolo11n.pt"
BASE_RES = 640
RES_LIST = [320]
EPOCHS = 25
BATCH = 32
RESULT_CSV = "/mnt/users/sdevkota/output/upscale_experiments_320_to_640_using_RealESRGAN/results_real_esrgan320to640.csv"

# Create result folder
os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
all_results = []

# Load Real-ESRGAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "/opt/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upscaler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

def copy_label(img_path, out_lbl_path):
    lbl_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    if os.path.exists(lbl_path):
        shutil.copy(lbl_path, out_lbl_path)

for res in RES_LIST:
    print(f"\n[RES {res}] Downscaling then upscaling using Real-ESRGAN")
    exp_dir = f"/mnt/users/sdevkota/output/upscale_experiments_320_to_640_using_RealESRGAN/train_from_{res}_to_{BASE_RES}"
    img_out_dir = os.path.join(exp_dir, "images")
    lbl_out_dir = os.path.join(exp_dir, "labels")

    for d in [img_out_dir, lbl_out_dir]:
        os.makedirs(d, exist_ok=True)

    total_sr_time = 0

    for subset, txt_file in zip(['train', 'val', 'test'], [SOURCE_TRAIN_TXT, SOURCE_VAL_TXT, SOURCE_TEST_TXT]):
        subset_img_dir = os.path.join(img_out_dir, subset)
        subset_lbl_dir = os.path.join(lbl_out_dir, subset)
        os.makedirs(subset_img_dir, exist_ok=True)
        os.makedirs(subset_lbl_dir, exist_ok=True)

        with open(txt_file, "r") as f:
            img_paths = [line.strip() for line in f if line.strip().endswith(".jpg")]

        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            sr_start_time = time()
            try:
                small = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
                img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                output, _ = upscaler.enhance(img_rgb, outscale=1)
                upscaled = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Real-ESRGAN failed for {img_path}: {e}")
                continue

            sr_total_time = time() - sr_start_time
            total_sr_time += sr_total_time

            out_img_path = os.path.join(subset_img_dir, os.path.basename(img_path))
            cv2.imwrite(out_img_path, upscaled)

            out_lbl_path = os.path.join(subset_lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
            copy_label(img_path, out_lbl_path)

    # Write data.yaml
    data_yaml_path = os.path.join(exp_dir, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {exp_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names: ['Normal sperm', 'Sperm cluster', 'Small or pinhead sperm']\n")

    # Train YOLO
    start_time = time()
    model = YOLO(MODEL_AR)
    model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=BASE_RES,
        batch=BATCH,
        project=exp_dir,
        name="yolo_model",
        patience=10
    )

    # Validation and Testing
    best_model_path = os.path.join(exp_dir, "yolo_model", "weights", "best.pt")
    trained_model = YOLO(best_model_path)

    val_metrics = trained_model.val(data=data_yaml_path, imgsz=BASE_RES, batch=16, split='val')
    test_metrics = trained_model.val(data=data_yaml_path, imgsz=BASE_RES, batch=16, split='test')

    all_results.append({
        "Downscaled_from": res,
        "Eval_Res": BASE_RES,
        "val_precision": val_metrics.box.mp,
        "val_Recall": val_metrics.box.mr,
        "val_mAP50": val_metrics.box.map50,
        "val_mAP50-95": val_metrics.box.map,
        "test_Precision": test_metrics.box.mp,
        "test_Recall": test_metrics.box.mr,
        "test_mAP50": test_metrics.box.map50,
        "test_mAP50-95": test_metrics.box.map,
        "training_time": round(time() - start_time, 2),
        "SR_processing_time": round(total_sr_time, 2)
    })

# Save results
df = pd.DataFrame(all_results)
df.to_csv(RESULT_CSV, index=False)
print(f"\n✔ Experiment finished. Results saved to {RESULT_CSV}")
