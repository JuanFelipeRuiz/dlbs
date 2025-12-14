'''
usage on one image:
    python infer.py --model runs/segment/yolo-seg-baseline-no-aug/weights/best.pt \
                    --image path/to/your/image.jpg

usage with path:
    python infer.py --model runs/segment/yolo-seg-baseline-no-aug/weights/best.pt \
                    --image path/to/your/image.jpg \
                    --out outputs/segmentation_result.jpg
'''
import argparse
from ultralytics import YOLO
import cv2


def run_inference(model_path: str, image_path: str, out_path: str = "segmentation_result.jpg"):
    model = YOLO(model_path)
    results = model(image_path)

    for r in results:
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            print(f"Gefunden: {len(masks)} Objekte")

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()

            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                print(f"Objekt {i+1}: Klasse={int(cls)}, Confidence={conf:.2f}")

        annotated = r.plot()
        cv2.imwrite(out_path, annotated)

    print(f"✓ Inferenz abgeschlossen, Ergebnis gespeichert: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Pfad zu best.pt")
    ap.add_argument("--image", required=True, help="Bildpfad")
    ap.add_argument("--out", default="segmentation_result.jpg")
    args = ap.parse_args()

    run_inference(args.model, args.image, args.out)
