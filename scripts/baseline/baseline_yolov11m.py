from ultralytics import YOLO
from time import time
import os
import yaml
import pandas as pd

# Configuration for YOLO MODEL
Yolo_MODELS = ['yolo11m.pt']
IMAGE_SIZE = 640
EPOCHS = 25
BATCH_SIZE = 64
NR_CPU_CORES = 16

DATA_YAML_PATH = '/mnt/users/sdevkota/data_yolo_corrected.yaml'
EXPERIMENT_BASE_DIR = '/mnt/users/sdevkota/output/yolo_640_experiment1_corrected_yolo11m_new'
RESULTS_CSV = os.path.join(EXPERIMENT_BASE_DIR, 'summary_results_corrected_yolo11m.csv')

all_results = []

for model_name in Yolo_MODELS:
    start_time = time()

    print(f"\n--- Training {model_name} ---")
    
    # Initializing model
    model = YOLO(model_name)

    # Definining output directory
    model_output_dir = os.path.join(EXPERIMENT_BASE_DIR, model_name.split('.')[0])

    # Training the model
    model.train(data=DATA_YAML_PATH,
                epochs=EPOCHS,
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                workers=NR_CPU_CORES,
                project=model_output_dir,
                patience=15)
    
    # Evaluating the trained model
    val_metrics = model.val(data=DATA_YAML_PATH,
                        imgsz=IMAGE_SIZE,
                        batch=BATCH_SIZE,
                        split='val')
    
    test_metrics = model.val(data=DATA_YAML_PATH,
                        imgsz=IMAGE_SIZE,
                        batch=BATCH_SIZE,
                        split='test')
    
    # Collecting the results
    result = {
        'model': model_name,
        'val_precision': val_metrics.results_dict.get('metrics/precision(B)', 0),
        'val_recall': val_metrics.results_dict.get('metrics/recall(B)', 0),
        'val_mAP50': val_metrics.results_dict.get('metrics/mAP50(B)', 0),
        'val_mAP50-95': val_metrics.results_dict.get('metrics/mAP50-95(B)', 0),
        'test_precision': test_metrics.results_dict.get('metrics/precision(B)', 0),
        'test_recall': test_metrics.results_dict.get('metrics/recall(B)', 0),
        'test_mAP50': test_metrics.results_dict.get('metrics/mAP50(B)', 0),
        'test_mAP50-95': test_metrics.results_dict.get('metrics/mAP50-95(B)', 0),
        'training_time_sec': round(time() - start_time, 2)
    }
    all_results.append(result)


df = pd.DataFrame(all_results)
df.to_csv(RESULTS_CSV, index=False)
print(f"\n Experiment concluded and results saved to {RESULTS_CSV}")
print(df)
