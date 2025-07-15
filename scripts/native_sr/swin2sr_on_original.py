import os
import cv2
import shutil
import pandas as pd
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
from time import time
from transformers import pipeline

# Configuration for the pipeline
SOURCE_TRAIN_TXT = "/mnt/users/sdevkota/train_corrected.txt"
SOURCE_VAL_TXT = "/mnt/users/sdevkota/val_corrected.txt"
SOURCE_TEST_TXT = "/mnt/users/sdevkota/test_yolo_m.txt"

MODEL_AR = "yolo11n.pt"
BASE_RES = 640
EPOCHS = 25
BATCH = 32
RESULT_CSV = "/mnt/users/sdevkota/output/original_with_SR_only/results_sr_only_from_original.csv"
EXP_DIR = "/mnt/users/sdevkota/output/original_with_SR_only/train_from_original_to_640"

os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
img_out_dir = os.path.join(EXP_DIR, "images")
lbl_out_dir = os.path.join(EXP_DIR, "labels")

# Hugging Face pipeline
hg_pipeline = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x4-64", device=0 if torch.cuda.is_available() else -1)

def copy_label(img_path, out_lbl_path):
    lbl_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    if os.path.exists(lbl_path):
        shutil.copy(lbl_path, out_lbl_path)

total_sr_time = 0
all_results = []

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
        orig_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        try:
            hg_result = hg_pipeline(images=orig_pil)
            if isinstance(hg_result, Image.Image):
                upscaled = cv2.cvtColor(np.array(hg_result), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"SR failed for {img_path}: {e}")
            continue

        sr_total_time = time() - sr_start_time
        total_sr_time += sr_total_time

        out_img_path = os.path.join(subset_img_dir, os.path.basename(img_path))
        cv2.imwrite(out_img_path, upscaled)

        out_lbl_path = os.path.join(subset_lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        copy_label(img_path, out_lbl_path)

# Writing data.yaml
data_yaml_path = os.path.join(EXP_DIR, "data.yaml")
with open(data_yaml_path, "w") as f:
    f.write(f"path: {EXP_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("names: ['Normal sperm', 'Sperm cluster', 'Small or pinhead sperm']\n")

# Training YOLO model
start_time = time()
model = YOLO(MODEL_AR)
model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=BASE_RES,
    batch=BATCH,
    project=EXP_DIR,
    name="yolo_model",
    patience=10
)

# Validation
best_model_path = os.path.join(EXP_DIR, "yolo_model", "weights", "best.pt")
trained_model = YOLO(best_model_path)
val_metrics = trained_model.val(data=data_yaml_path, imgsz=BASE_RES, batch=16, split='val')

# Testing
test_metrics = trained_model.val(data=data_yaml_path, imgsz=BASE_RES, batch=16, split='test')

# Save results
all_results.append({
    "Downscaled_from": "Original",
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

df = pd.DataFrame(all_results)
df.to_csv(RESULT_CSV, index=False)
print(f"\n Experiment Concluded and all results saved to {RESULT_CSV}")
