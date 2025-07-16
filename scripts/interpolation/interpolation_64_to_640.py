import os
import cv2
import shutil
import pandas as pd
import numpy as np
from ultralytics import YOLO
from time import time
import albumentations as A

# configuration for the pipeline
SOURCE_TRAIN_TXT = "/mnt/users/sdevkota/train_corrected.txt"
SOURCE_VAL_TXT   = "/mnt/users/sdevkota/val_corrected.txt"
SOURCE_TEST_TXT  = "/mnt/users/sdevkota/test_yolo_m.txt"

MODEL_AR = "yolo11n.pt"
BASE_RES = 640
RES_LIST = [64]
EPOCHS = 25
BATCH = 32
RESULT_CSV = "/mnt/users/sdevkota/output/interpolation_pipeline/results_64_to_640.csv"

os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
all_results = []

# Albumentations transformation
bbox_transform = A.Compose([
    A.Resize(64, 64),
    A.Resize(640, 640)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

for res in RES_LIST:
    exp_dir = f"/mnt/users/sdevkota/output/interpolation_pipeline/train_from_{res}_to_{BASE_RES}"
    img_out_dir = os.path.join(exp_dir, "images")
    lbl_out_dir = os.path.join(exp_dir, "labels")

    for d in [img_out_dir, lbl_out_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n[INFO] Processing images at {res}x{res} → {BASE_RES}x{BASE_RES}")
    processing_start = time()

    for subset, txt_file in zip(["train", "val", "test"], [SOURCE_TRAIN_TXT, SOURCE_VAL_TXT, SOURCE_TEST_TXT]):
        subset_img_dir = os.path.join(img_out_dir, subset)
        subset_lbl_dir = os.path.join(lbl_out_dir, subset)
        os.makedirs(subset_img_dir, exist_ok=True)
        os.makedirs(subset_lbl_dir, exist_ok=True)

        with open(txt_file, "r") as f:
            img_paths = [line.strip() for line in f if line.strip().endswith(".jpg")]

        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Skipping unreadable image: {img_path}")
                continue

            
            small = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
            upscaled = cv2.resize(small, (BASE_RES, BASE_RES), interpolation=cv2.INTER_LINEAR)

            
            out_img_path = os.path.join(subset_img_dir, os.path.basename(img_path))
            cv2.imwrite(out_img_path, upscaled)

            
            label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
            out_lbl_path = os.path.join(subset_lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
            if not os.path.exists(label_path):
                continue

            bboxes = []
            classes = []
            with open(label_path, "r") as fin:
                for line in fin:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = map(float, parts)
                    bboxes.append([x, y, w, h])
                    classes.append(int(cls))

            if not bboxes:
                continue

            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                result = bbox_transform(image=dummy_img, bboxes=bboxes, class_labels=classes)
                with open(out_lbl_path, "w") as fout:
                    for cls, (x, y, w, h) in zip(result["class_labels"], result["bboxes"]):
                        fout.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            except Exception as e:
                print(f"[BBox FAIL] {label_path}: {e}")

    processing_end = time()
    processing_duration = round(processing_end - processing_start, 2)

    # Writing data.yaml file
    data_yaml_path = os.path.join(exp_dir, "data.yaml")
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {exp_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names: ['Normal sperm', 'Sperm cluster', 'Small or pinhead sperm']\n")

    # Training YOLO Model
    model = YOLO(MODEL_AR)
    train_start = time()
    model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=BASE_RES,
        batch=BATCH,
        project=exp_dir,
        name="yolo_model",
        patience=10
    )
    train_end = time()
    training_duration = round(train_end - train_start, 2)

    # Evaluating the model
    best_model_path = os.path.join(exp_dir, "yolo_model", "weights", "best.pt")
    trained_model = YOLO(best_model_path)

    val_start = time()
    val_metrics = trained_model.val(data=data_yaml_path, imgsz=BASE_RES, batch=16, split="val")
    val_end = time()

    test_start = time()
    test_metrics = trained_model.val(data=data_yaml_path, imgsz=BASE_RES, batch=16, split="test")
    test_end = time()

    val_time = round(val_end - val_start, 2)
    test_time = round(test_end - test_start, 2)

    
    all_results.append({
        "Downscaled_from": res,
        "Eval_Res": BASE_RES,
        "val_precision": val_metrics.box.mp,
        "val_recall": val_metrics.box.mr,
        "val_mAP50": val_metrics.box.map50,
        "val_mAP50-95": val_metrics.box.map,
        "test_precision": test_metrics.box.mp,
        "test_recall": test_metrics.box.mr,
        "test_mAP50": test_metrics.box.map50,
        "test_mAP50-95": test_metrics.box.map,
        "image_processing_time_sec": processing_duration,
        "yolo_training_time_sec": training_duration,
        "val_inference_time_sec": val_time,
        "test_inference_time_sec": test_time
    })

# Saving all results
df = pd.DataFrame(all_results)
df.to_csv(RESULT_CSV, index=False)
print(f"\nExperiment concluded and Results saved to: {RESULT_CSV}")
