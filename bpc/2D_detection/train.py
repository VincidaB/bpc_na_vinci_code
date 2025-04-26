import os
from ultralytics import YOLO
import torch
import argparse

def train_yolo11(task, data_path, obj_id, epochs, imgsz, batch, resume=False, model_path=None):
    """
    Train YOLO11 for a specific task ("detection" or "segmentation")
    using Ultralytics YOLO with a single object class.

    Args:
        task (str): "detection" or "segmentation"
        data_path (str): Path to the YOLO .yaml file (e.g. data_obj_11.yaml).
        obj_id (int): The BOP object ID (e.g. 11).
        epochs (int): Number of training epochs.
        imgsz (int): Image size used for training.
        batch (int): Batch size.
        resume (bool): Whether to resume the training.
        model_path (str): Path to the previously trained model .pt file.

    Returns:
        final_model_path (str): Path where the trained model is saved.
    """

    # Decide the device automatically: MPS (Apple), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Pick the pre-trained model file based on task
    if task == "detection":
        if not resume:
            print("Starting training from basic pre-trained model yolo11n.pt.")
            pretrained_weights = "yolo11n.pt"
        elif not os.path.exists(model_path):
            print(f"Couldn't find specified model at {model_path}. Starting training from basic pre-trained model yolo11n.pt.")
            pretrained_weights = "yolo11n.pt"
        else:
            print(f"Resuming training from {model_path}.")
            pretrained_weights = model_path
        task_suffix = "detection"
    elif task == "segmentation":
        pretrained_weights = "yolo11n-seg.pt"
        task_suffix = "segmentation"
    else:
        print("Invalid task. Must be 'detection' or 'segmentation'.")
        return None

    # Check if the dataset YAML file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset YAML file not found at {data_path}")
        return None

    # Load the YOLO model
    print(f"Loading model {pretrained_weights} for {task_suffix} ...")
    model = YOLO(pretrained_weights)
    if resume:
        model.resume = True

    # Train the model
    print(f"Training YOLO11 for {task_suffix} on object {obj_id} using {device}...")
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=8,  # more workers, train faster, adjust based on device
        save=True,  # This creates a 'runs/train/...' folder but we'll still save final .pt ourselves
        # resume=resume,
        # patience=20,
        cache=True,
    )

    # ----------------------------------------------------------------------------
    # Force the final save to your desired path:
    #   idp_codebase/yolo/models/<detection or segmentation>/obj_<obj_id>/yolo11-<task_suffix>-obj_<obj_id>.pt
    # ----------------------------------------------------------------------------
    save_dir = os.path.join("bpc","yolo", "models", task_suffix, f"obj_{obj_id}")
    os.makedirs(save_dir, exist_ok=True)

    model_name = f"yolo11-{task_suffix}-obj_{obj_id}.pt"
    final_model_path = os.path.join(save_dir, model_name)

    # Save final model
    model.save(final_model_path)

    print(f"Model saved as: {final_model_path}")
    return final_model_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on a specific dataset and object.")
    parser.add_argument("--obj_id", type=int, required=True, help="Object ID for training (e.g., 18).")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the dataset YAML file (e.g. bpc/2D_detection/configs/data_obj_18.yaml).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--task", type=str, choices=["detection", "segmentation"], default="detection",
                        help="Task type (detection or segmentation).")
    parser.add_argument("--resume", type=bool, default=False,
                        help="Whether to resume from a previously trained model. Please specify --model_path if True.")
    parser.add_argument("--model_path", type=str, default="",
                        help="If --resume is True, path to the model to build upon (e.g. runs/detect/train/weights/best.pt).")

    args = parser.parse_args()

    train_yolo11(
        task=args.task,
        data_path=args.data_path,
        obj_id=args.obj_id,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        resume=args.resume,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()
