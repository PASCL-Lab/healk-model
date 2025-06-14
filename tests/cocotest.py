# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from torch import tensor
# from torchmetrics.detection import MeanAveragePrecision
# from pprint import pprint

# # Load ground-truth annotations and predictions in COCO format
# coco_gt = COCO("ground_truth.json")
# coco_dt = coco_gt.loadRes("predictions.json")

# # Run evaluation
# coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()

from torchmetrics.detection import MeanAveragePrecision
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pprint import pprint


#  metric = MeanAveragePrecision(iou_type="bbox")
#  metric.update(preds, target)
#  pprint(metric.compute())



model_predictions = [
    {
        "boxes": torch.tensor([[74.4506, 117.9982, 400.6461, 430.6247],
                               [51.4281, 64.0308, 432.2385, 455.4966],
                               [65.5099, 106.5663, 413.3722, 434.6424],
                               [54.6173, 70.6974, 426.3994, 422.5687]]),
        "labels": torch.tensor([17, 18, 34, 30]),
        "scores": torch.tensor([0.0817, 0.0723, 0.0651, 0.0543]),
    }
]

# Ground truth (requires reformatting)
ground_truths = [
    {
        "boxes": torch.tensor([[206.6032, 392.6349, 288.3810, 439.8730],
                               [87.8730, 353.6931, 157.7143, 407.0265]]),
        "labels": torch.tensor([0, 0]),  # Replace class_labels with labels
    }
]

print("DONE APPENDING")
metric = MeanAveragePrecision(iou_type="bbox")
metric.update(model_predictions, ground_truths)
pprint(metric.compute())


mAP = average_precision_score(ground_truths, model_predictions)
print(f"Mean Average Precision: {mAP:.4f}")


({'boxes': tensor([[206.6032, 392.6349, 288.3810, 439.8730],
        [ 87.8730, 353.6931, 157.7143, 407.0265]]), 'labels': tensor([0, 0]), 'class_labels': [0, 0], 'area': tensor([3863.0264, 3724.8682]), 'iscrowd': tensor([0, 0]), 'image_id': tensor([0]), 'image_name': 'almond_10.webp'},)


  from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

# Initialize the Mean Average Precision metric
metric = MeanAveragePrecision()

# Placeholders for predictions and ground truths
model_predictions = []
ground_truths = []

# Loop through the validation data
for i, data in enumerate(prog_bar):
    images, targets = data

    with torch.no_grad():
        # Get predictions from the model
        predictions = model(images, targets)

    # Process model predictions (make sure they contain 'boxes', 'labels', 'scores')
    formatted_predictions = []
    for item in predictions:
        formatted_predictions.append({
            'boxes': item['boxes'],  # Bounding boxes
            'labels': item['labels'],  # Predicted labels
            'scores': item['scores']  # Prediction scores
        })
    
    # Process ground truth data (keep only 'boxes' and 'labels')
    formatted_targets = []
    for target in targets:
        formatted_targets.append({
            'boxes': target['boxes'],  # Ground truth bounding boxes
            'labels': target['labels']  # Ground truth labels
        })

    # Append the processed predictions and targets to the lists
    model_predictions.append(formatted_predictions)
    ground_truths.append(formatted_targets)

# Now use the metric to compute the Mean Average Precision
metric.update(model_predictions, ground_truths)

# Compute the metric
results = metric.compute()

# Print results
print(results)
