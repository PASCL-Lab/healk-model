import torch
from tqdm.auto import tqdm
from torchmetrics.detection import MeanAveragePrecision
from config import DEVICE, NUM_WORKERS, CLASSES
from datasets import create_valid_dataset, create_valid_loader
from model import create_model
from pprint import pprint

num_classes = len(CLASSES)


def validate(valid_loader, model, device):
    model.to(device).eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(valid_loader, total=len(valid_loader), desc="Validating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            outputs = model(images)

            # Move outputs and targets to CPU and format them for torchmetrics
            outputs = [
                {
                    "boxes": o["boxes"].cpu(),
                    "scores": o["scores"].cpu(),
                    "labels": o["labels"].cpu()
                }
                for o in outputs
            ]
            targets = [
                {
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu()
                }
                for t in targets
            ]

            metric.update(outputs, targets)

    return metric.compute()


def run_validator():
    print("Loading validation data...")
    valid_dataset = create_valid_dataset()
    # Keep batch size 1 for evaluation
    valid_loader = create_valid_loader(valid_dataset)

    print(f"Number of validation samples: {len(valid_dataset)}\n")

    print("Loading model...")
    model = create_model(num_classes=num_classes)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Starting validation...")
    metrics = validate(valid_loader, model, DEVICE)

    print("\nValidation Results:")
    pprint(metrics)


if __name__ == "__main__":
    run_validator()
