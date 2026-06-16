'''
usage:
    python validate.py  --model runs/segment/yolo-seg-baseline-no-aug/weights/best.pt \
                        --data dataset.yaml
'''

import argparse
from ultralytics import YOLO


def validate_model(model_path: str, data_yaml: str = "dataset.yaml"):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    # Box mAP is also available for segmentation; segmentation metrics may be under metrics.seg depending on the version.
    print("\nValidation results (Box):")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")

    if hasattr(metrics, "seg") and metrics.seg is not None:
        print("\nValidation results (Seg):")
        print(f"mAP50:     {metrics.seg.map50:.3f}")
        print(f"mAP50-95:  {metrics.seg.map:.3f}")

    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best.pt")
    ap.add_argument("--data", default="dataset.yaml", help="dataset.yaml")
    args = ap.parse_args()

    validate_model(args.model, args.data)
