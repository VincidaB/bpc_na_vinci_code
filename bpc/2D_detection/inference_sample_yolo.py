import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import argparse


def show_example_detection(model_path, image_path):
    # Load YOLO model
    model = YOLO(model_path)

    # Load image using OpenCV
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB (Matplotlib expects RGB format)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(img_rgb)[0]

    # Draw detections directly on the image
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Draw bounding box in blue

    # Display the image using Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")  # Hide axis
    plt.title("YOLO Detection")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on a specific dataset and object.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to your trained mode (e.g. `bpc/yolo/models/detection/obj_18/yolo11-detection-obj_18.pt`)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="One image to assess your model on (e.g. `datasets/data/train_pbr/000005/rgb_cam1/000001.jpg`)")
    
    args = parser.parse_args()

    show_example_detection(
        model_path=args.model_path,
        image_path=args.image_path
    )


if __name__ == "__main__":
    main()