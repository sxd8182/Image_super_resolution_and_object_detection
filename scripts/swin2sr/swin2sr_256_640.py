import os
import time
import cv2
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
import torch
import albumentations as A

# === Central Config ===
CONFIG = {
    "source_txt": {
        "train": "/mnt/users/sdevkota/train_corrected.txt",
        "val": "/mnt/users/sdevkota/val_corrected.txt",
        "test": "/mnt/users/sdevkota/test_yolo_m.txt"
    },
    "base_res": 640,
    "res_list": [256],
    "epochs": 25,
    "batch": 32,
    "model_path": "yolo11n.pt",
    "output_dir": "/mnt/users/sdevkota/output/swin2sr_pipeline",
    "sr_model": "caidas/swin2SR-classical-sr-x4-64",
    "class_names": ['Normal sperm', 'Sperm cluster', 'Small or pinhead sperm']
}

DEVICE = 0 if torch.cuda.is_available() else -1
hg_pipeline = pipeline("image-to-image", model=CONFIG["sr_model"], device=DEVICE)

# bounding box transformations
bbox_transform = A.Compose([
    A.Resize(256, 256),
    A.Resize(1056, 1056),
    A.Resize(640, 640)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# helper functions
def load_image_paths(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f if line.strip().endswith(".jpg")]

def run_sr(img_path, res):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Warning] Could not read image: {img_path}")
        return None
    small = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    small_pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    try:
        result = hg_pipeline(images=small_pil)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR) if isinstance(result, Image.Image) else None
    except Exception as e:
        print(f"[SR FAIL] {img_path}: {e}")
        return None

def transform_and_save_labels(label_path, out_lbl_path):
    if not os.path.exists(label_path):
        return
    bboxes, classes = [], []
    with open(label_path, "r") as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            bboxes.append([x, y, w, h])
            classes.append(int(cls))
    if not bboxes:
        return
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        result = bbox_transform(image=dummy_img, bboxes=bboxes, class_labels=classes)
        with open(out_lbl_path, "w") as fout:
            for cls, (x, y, w, h) in zip(result["class_labels"], result["bboxes"]):
                fout.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    except Exception as e:
        print(f"[BBox Fail] {label_path}: {e}")

def resize_images_to_640(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(".jpg"):
            img = cv2.imread(os.path.join(input_dir, fname))
            if img is not None:
                resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(output_dir, fname), resized)

# main pipeline
all_results = []
for res in CONFIG["res_list"]:
    exp_dir = os.path.join(CONFIG["output_dir"], f"train_from_{res}_to_{CONFIG['base_res']}")
    img_out_dir = os.path.join(exp_dir, "images")
    lbl_out_dir = os.path.join(exp_dir, "labels")
    for d in [img_out_dir, lbl_out_dir]:
        os.makedirs(d, exist_ok=True)

    sr_start_time = time.time()
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(img_out_dir, split)
        lbl_dir = os.path.join(lbl_out_dir, split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        img_paths = load_image_paths(CONFIG["source_txt"][split])
        for img_path in img_paths:
            upscaled = run_sr(img_path, res)
            if upscaled is None:
                continue

            out_img_path = os.path.join(img_dir, os.path.basename(img_path))
            cv2.imwrite(out_img_path, upscaled)

            label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
            out_lbl_path = os.path.join(lbl_dir, os.path.basename(label_path))
            transform_and_save_labels(label_path, out_lbl_path)

    sr_duration = time.time() - sr_start_time

    # Resizing to 640x640
    final_img_dir = os.path.join(exp_dir, "images_resized_640")
    for split in ["train", "val", "test"]:
        resize_images_to_640(os.path.join(img_out_dir, split), os.path.join(final_img_dir, split))

    # Saving data.yaml
    data_yaml = os.path.join(exp_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {exp_dir}\n")
        f.write("train: images_resized_640/train\n")
        f.write("val: images_resized_640/val\n")
        f.write("test: images_resized_640/test\n")
        f.write(f"names: {CONFIG['class_names']}\n")

    # Training YOLO
    model = YOLO(CONFIG["model_path"])
    train_start = time.time()
    model.train(
        data=data_yaml,
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["base_res"],
        batch=CONFIG["batch"],
        project=exp_dir,
        name="yolo_model",
        patience=10
    )
    train_duration = time.time() - train_start

    # Evaluating
    trained_model = YOLO(os.path.join(exp_dir, "yolo_model", "weights", "best.pt"))

    def evaluate(model, split):
        start = time.time()
        metrics = model.val(data=data_yaml, imgsz=CONFIG["base_res"], batch=16, split=split)
        return metrics, time.time() - start

    val_metrics, val_duration = evaluate(trained_model, "val")
    test_metrics, test_duration = evaluate(trained_model, "test")

    all_results.append({
        "Downscaled_from": res,
        "Eval_Res": CONFIG["base_res"],
        "val_precision": val_metrics.box.mp,
        "val_recall": val_metrics.box.mr,
        "val_mAP50": val_metrics.box.map50,
        "val_mAP50-95": val_metrics.box.map,
        "test_precision": test_metrics.box.mp,
        "test_recall": test_metrics.box.mr,
        "test_mAP50": test_metrics.box.map50,
        "test_mAP50-95": test_metrics.box.map,
        "sr_processing_time_sec": sr_duration,
        "yolo_training_time_sec": train_duration,
        "val_inference_time_sec": val_duration,
        "test_inference_time_sec": test_duration
    })

# saving final outputs
results_csv = os.path.join(CONFIG["output_dir"], "results.csv")
pd.DataFrame(all_results).to_csv(results_csv, index=False)
print(f"\nFull pipeline complete. Results saved to: {results_csv}")
