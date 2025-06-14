import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from config import DEVICE, NUM_WORKERS
from datasets import create_train_dataset, create_train_loader, create_valid_dataset, create_valid_loader
from utils import get_valid_transform
from tqdm.auto import tqdm
from model import create_model
import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from config import CLASSES
from sklearn.metrics import average_precision_score

# CLASSES = [
#     # 'background',
#     "almond",
#     "apple",
#     "avocado",
#     "banana",
#     "beetroot",
#     "bell pepper",
#     "blueberries",
#     "broccoli",
#     "brussels",
#     "cabbage",
#     "carrot",
#     "cauliflower",
#     "chilli",
#     "corn",
#     "cucumber",
#     "eggplant",
#     "garlic",
#     "ginger",
#     "grapes",
#     "jalepeno",
#     "kiwi",
#     "lemon",
#     "lettuce",
#     "mango",
#     "onion",
#     "orange",
#     "paprika",
#     "parsley",
#     "pear",
#     "peas",
#     "pineapple",
#     "pomegranate",
#     "potato",
#     "raddish",
#     "soybeans",
#     "spinach",
#     "strawberries",
#     "tomato",
#     "turnip",
#     "walnut",
#     "watermelon"
# ]
num_classes = len(CLASSES)

# def calculate_iou(box1, box2):
#     """
#     Calculate Intersection over Union (IoU) for two bounding boxes.
#     """
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     # Compute intersection area
#     inter_area = max(0, x2 - x1) * max(0, y2 - y1)

#     # Compute union area
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area if union_area > 0 else 0

# def evaluate_model(predictions, ground_truths, iou_threshold=0.5):
#     """
#     Evaluate a model's predictions to compute mAP and AR.
#     Args:
#         predictions: List of dictionaries, each with 'boxes', 'scores', and 'labels'.
#         ground_truths: List of dictionaries, each with 'boxes' and 'labels'.
#         iou_threshold: IoU threshold for considering a detection as TP.
#     Returns:
#         mean_ap: Mean Average Precision (mAP).
#         average_recall: Average Recall (AR).
#     """
#     all_precisions = []
#     all_recalls = []
#     all_aps = []

#     for class_id in range(1, num_classes + 1):  # Assume class 0 is background
#         tp = 0
#         fp = 0
#         fn = 0
#         precisions = []
#         recalls = []

#         for preds, gts in zip(predictions, ground_truths):
#             pred_boxes = [box for box, label in zip(preds['boxes'], preds['labels']) if label == class_id]
#             gt_boxes = [box for box, label in zip(gts['boxes'], gts['labels']) if label == class_id]

#             matched_gt = set()
#             for pred_box in pred_boxes:
#                 ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
#                 best_iou = max(ious) if ious else 0

#                 if best_iou >= iou_threshold:
#                     tp += 1
#                     matched_gt.add(gt_boxes[ious.index(best_iou)])
#                 else:
#                     fp += 1

#             fn += len(gt_boxes) - len(matched_gt)

#         # Compute precision and recall for this class
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         precisions.append(precision)
#         recalls.append(recall)

#         # Compute AP (average precision)
#         ap = np.mean(precisions) if precisions else 0
#         all_aps.append(ap)

#     # Compute mAP and AR
#     mean_ap = np.mean(all_aps)
#     average_recall = np.mean(recalls)

#     return mean_ap, average_recall


# # Example usage
# predictions = [
#     {'boxes': [[50, 50, 150, 150]], 'scores': [0.9], 'labels': [1]},  # Predicted boxes
# ]
# ground_truths = [
#     {'boxes': [[48, 48, 152, 152]], 'labels': [1]},  # Ground-truth boxes
# ]

# num_classes = 1  # Set the number of classes in your dataset
# mAP, AR = evaluate_model(predictions, ground_truths)
# print(f"mAP: {mAP:.4f}, AR: {AR:.4f}")


def iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1: (tensor) Bounding box 1, tensor of shape [4] (xmin, ymin, xmax, ymax).
        box2: (tensor) Bounding box 2, tensor of shape [4] (xmin, ymin, xmax, ymax).
    Returns:
        IoU value (float).
    """
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Compute intersection
    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate precision and recall based on predicted boxes and ground truths.
    Args:
        predictions: List of dictionaries containing predicted boxes and scores.
        ground_truths: List of dictionaries containing ground truth boxes and labels.
        iou_threshold: IoU threshold to consider a prediction as valid (default: 0.5).
    Returns:
        precision, recall
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Go through each prediction
    for prediction in predictions:
        boxes_pred = prediction['boxes']
        labels_pred = prediction['labels']
        scores_pred = prediction['scores']

        # Go through each ground truth
        for gt in ground_truths:
            boxes_gt = gt['bbox']
            labels_gt = gt['label']

            for i, box_pred in enumerate(boxes_pred):
                # Check if the predicted label matches the ground truth label
                if labels_pred[i] == labels_gt:
                    # Compute IoU
                    iou_score = iou(box_pred, boxes_gt)

                    # If IoU is above threshold, consider it a true positive
                    if iou_score >= iou_threshold:
                        true_positives += 1
                    else:
                        false_positives += 1

            # If ground truth object not detected
            false_negatives += len(boxes_gt) - true_positives

    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0

    return precision, recall


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate mean average precision (mAP).
    Args:
        predictions: List of dictionaries containing predicted boxes and scores.
        ground_truths: List of dictionaries containing ground truth boxes and labels.
        iou_threshold: IoU threshold to consider a prediction as valid (default: 0.5).
    Returns:
        mAP value (float).
    """
    aps = []  # List to store average precision for each class

    # List of unique classes
    classes = list(set([gt['label'] for gt in ground_truths]))

    for c in classes:
        # Get predictions and ground truths for class `c`
        preds_class = [p for p in predictions if p['labels'] == c]
        gts_class = [gt for gt in ground_truths if gt['label'] == c]

        precision, recall = calculate_precision_recall(
            preds_class, gts_class, iou_threshold)

        # Calculate AP for the class using precision-recall curve (simplified here)
        # Example of a simple calculation, in practice you would use the PR curve
        ap = precision * recall
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0


def calculate_average_recall(predictions, ground_truths, iou_thresholds=[0.5, 0.75]):
    """
    Calculate average recall at different IoU thresholds.
    Args:
        predictions: List of dictionaries containing predicted boxes and scores.
        ground_truths: List of dictionaries containing ground truth boxes and labels.
        iou_thresholds: List of IoU thresholds for recall evaluation (default: [0.5, 0.75]).
    Returns:
        average recall value (float).
    """
    recalls = []

    for threshold in iou_thresholds:
        _, recall = calculate_precision_recall(
            predictions, ground_truths, threshold)
        recalls.append(recall)

    return sum(recalls) / len(recalls) if recalls else 0


all_preds = []
all_labels = []


predictions = []
ground_truths = []


def validate(valid_data_loader, model):
    # print('Validating')

    # # initialize tqdm progress bar
    # # prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    # for images, targets in valid_data_loader:

    #     images = list(image.to(DEVICE) for image in images)
    #     targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

    #     with torch.no_grad():
    #         loss_dict = model(images, targets)

    #     print(loss_dict, "THE DICT")
    #     # losses = sum(loss for loss in loss_dict.values())
    #     # loss_value = losses.item()
    #     # val_loss_list.append(loss_value)
    #     # val_loss_hist.send(loss_value)
    #     # val_itr += 1
    #     # # update the loss value beside the progress bar for each iteration
    #     # prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    # return val_loss_list
    for images, targets in valid_data_loader:
        # Get predictions
        predictions = model(images, targets)
        pred_boxes = predictions[0]['boxes']  # Predicted bounding boxes
        pred_labels = predictions[0]['labels']  # Predicted labels
        pred_scores = predictions[0]['scores']  # Prediction scores

        # print(predictions, "ACE")

        # print("##########################")
        # print(targets, "LABEL")
        # print("##########################")
        # Ground truth boxes and labels
        # true_boxes = targets[0]['annotation']['object']  # Assuming VOC format
        # true_labels = [obj['name'] for obj in true_boxes]
        # Bounding boxes in (x_min, y_min, x_max, y_max) format
        true_boxes = targets[0]['boxes']
        true_labels = targets[0]['labels']

        # Map ground truth labels to class indices
        # true_labels = [CLASSES.index(label) for label in true_labels]

        # Apply IoU threshold and calculate mAP for each class (or overall)
        # for idx in range(len(true_labels)):
        #     label = pred_labels[idx]
        #     if label.item() in true_labels:
        #         iou = box_iou(pred_boxes, true_boxes)
        #         # Check if IoU > threshold (0.5 is common for detection tasks)
        #         # print("THE IOU", iou)
        #         # if iou > 0.5:
        #         #     all_preds.append(pred_labels[idx].item())
        #         #     all_labels.append(true_labels[idx])
        #         print("IOU",iou)
        #         for i in range(iou.size(0)):
        #             if iou[i].item() > 0.5:  # Compare the IoU value to the threshold
        #                 all_preds.append(pred_labels[idx].item())
        #                 all_labels.append(true_labels[i].item())

        for idx in range(len(true_labels)):
           # Check if the predicted label corresponds to the current true label
            if pred_labels[idx].item() == true_labels[idx].item():
                # Compute IoU between all predicted boxes and true boxes
                iou = box_iou(pred_boxes[idx].unsqueeze(
                    0), true_boxes[idx].unsqueeze(0))

                # Only consider predictions with IoU > 0.5
                if iou.item() > 0.5:
                    all_preds.append(pred_labels[idx].item())
                    all_labels.append(true_labels[idx].item())


def run_validator():
    valid_dataset = create_valid_dataset()
    valid_loader = create_valid_loader(valid_dataset, 0)
    print(f"Number of validation samples: {len(valid_dataset)}\n")
    # initialize the model and move to the computation device
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(
        'outputs/best_model.pth')['model_state_dict'])
    model.eval()

    model = model.to(DEVICE)
    # get the model parameters
    # params = [p for p in model.parameters() if p.requires_grad]
    # # define the optimizer
    # optimizer = torch.optim.SGD(
    #     params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    val_loss = validate(valid_loader, model)


run_validator()
# Assume predictions and ground_truths are loaded as you have provided in the input
precision, recall = calculate_precision_recall(all_preds, all_labels)
map_value = calculate_map(all_preds, all_labels)
avg_recall = calculate_average_recall(all_preds, all_labels)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"mAP: {map_value:.4f}")
print(f"Average Recall: {avg_recall:.4f}")


mAP = average_precision_score(all_labels, all_preds)
print(f"Mean Average Precision: {mAP:.4f}")
