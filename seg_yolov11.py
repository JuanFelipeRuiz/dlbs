"""
YOLO Segmentation Training Script
Verwendet YOLOv11 für Instance Segmentation
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import yaml
import sys
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# Dataset-Struktur erstellen
def create_dataset_structure():
    """Erstellt die benötigte Ordnerstruktur für YOLO"""
    dirs = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Dataset-Struktur erstellt")


def train_baseline_no_augmentation(epochs=50, batch_size=16,imgsz=640):
    """Trainiert BASELINE-Modell OHNE Augmentation"""
    # wandb initialisieren
    project_name = 'yolo-segmentation'
    name = 'yolo-seg-baseline-no-aug'

    wandb.init(
        project=project_name,
        name= name,
        config={
            'model': 'yolo11n-seg',
            'epochs': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'augmentation': 'none'
        }
    )

    model = YOLO('yolo11n-seg.pt')

    # W&B Callback hinzufügen (optional, für mehr Features)
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Training mit deaktivierten Augmentationen
    results = model.train(
        data='dataset.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        device='cuda',  # or 'cpu'
        patience=10,
        save=True,
        plots=True,
        name='baseline_no_aug',
        # ===== AUGMENTATIONEN DEAKTIVIEREN =====
        # Geometrische Transformationen
        degrees=0.0,  # Rotation
        translate=0.0,  # Translation
        scale=0.0,  # Skalierung
        shear=0.0,  # Scherung
        perspective=0.0,  # Perspektive
        flipud=0.0,  # Vertical Flip
        fliplr=0.0,  # Horizontal Flip
        # Farb-Augmentationen
        hsv_h=0.0,  # Hue
        hsv_s=0.0,  # Saturation
        hsv_v=0.0,  # Value/Brightness
        # Mix-Augmentationen
        mosaic=0.0,  # Mosaic (4 Bilder kombinieren)
        mixup=0.0,  # MixUp
        copy_paste=0.0,  # Copy-Paste (nur Segmentation)
        # Weitere Augmentationen
        erasing=0.0,  # Random Erasing
        auto_augment=None,  # AutoAugment/RandAugment deaktivieren
        # Mosaic früh schließen
        close_mosaic=0,  # Mosaic sofort deaktivieren
    )

    # W&B Run beenden
    wandb.finish()

    print("Baseline-Training (OHNE Augmentation) abgeschlossen")
    print(f"  Dashboard: https://wandb.ai/{wandb.run.entity}/{project_name}/runs/{wandb.run.id}")

    return model

#TODO: Ab hier weiterarbeiten / baseline testen ############################################################################
def train_yolo_segmentation():
    """Trainiert das YOLOv11 Segmentierungsmodell"""

    # Modell laden (n=nano, s=small, m=medium, l=large, x=xlarge)
    model = YOLO('yolo11n-seg.pt')  # nano-Version für schnelles Training

    # Training starten
    results = model.train(
        data='dataset.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        device='cuda',  # or 'cpu'
        patience=10,
        save=True,
        plots=True,
        name='yolo_segmentation'
    )

    print("Training abgeschlossen")
    return model

# Inferenz auf einem Bild durchführen
def run_inference(model_path='runs/segment/yolo_segmentation/weights/best.pt',
                  image_path='path/to/your/image.jpg'):
    """Führt Segmentierung auf einem Bild durch"""

    model = YOLO(model_path)

    # Vorhersage
    results = model(image_path)

    # Ergebnisse verarbeiten
    for r in results:
        # Segmentierungsmasken
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            print(f"Gefunden: {len(masks)} Objekte")

            # Bounding Boxes und Klassen
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                print(f"Objekt {i + 1}: Klasse={int(cls)}, Confidence={conf:.2f}")

        # Visualisierung speichern
        annotated = r.plot()
        cv2.imwrite('segmentation_result.jpg', annotated)

    print("✓ Inferenz abgeschlossen, Ergebnis gespeichert")


# 6. Modell validieren
def validate_model(model_path='runs/segment/yolo_segmentation/weights/best.pt'):
    """Validiert das trainierte Modell"""

    model = YOLO(model_path)
    metrics = model.val(data='dataset.yaml')

    print(f"\nValidierungsergebnisse:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    return metrics




if __name__ == "__main__":

    print("\n=== Training starten ===")
    baseline_model = train_baseline_no_augmentation()
    #model = train_yolo_segmentation()


    # Inferenz
    # run_inference(image_path='path/to/your/test_image.jpg')

    # Validierung
    # validate_model()