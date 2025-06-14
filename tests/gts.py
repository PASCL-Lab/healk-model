# import os
# import xml.etree.ElementTree as ET
# from config import CLASSES

# def parse_pascal_voc_annotation(xml_path):
#     """
#     Parse a Pascal VOC XML annotation file to extract bounding boxes and class labels.
#     Args:
#         xml_path: Path to the XML annotation file.
#     Returns:
#         ground_truths: List of dictionaries containing bounding boxes and labels.
#     """
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
    
#     ground_truths = []
    
#     for obj in root.findall('object'):
#         # Get the class label
#         class_label = obj.find('name').text
        
#         # Get the bounding box coordinates
#         bndbox = obj.find('bndbox')
#         xmin = int(bndbox.find('xmin').text)
#         ymin = int(bndbox.find('ymin').text)
#         xmax = int(bndbox.find('xmax').text)
#         ymax = int(bndbox.find('ymax').text)
        
#         # Store the bounding box and class label
#         ground_truths.append({
#             'bbox': [xmin, ymin, xmax, ymax],  # [x_min, y_min, x_max, y_max]
#             'label': class_label
#         })
    
#     return ground_truths

# def get_ground_truths_from_directory(annotation_dir):
#     """
#     Parse all Pascal VOC XML annotation files in a directory to extract ground truths.
#     Args:
#         annotation_dir: Directory containing Pascal VOC XML annotation files.
#     Returns:
#         all_ground_truths: List of dictionaries containing bounding boxes and labels for all images.
#     """
#     all_ground_truths = []
    
#     for xml_file in os.listdir(annotation_dir):
#         if xml_file.endswith('.xml'):
#             xml_path = os.path.join(annotation_dir, xml_file)
#             ground_truths = parse_pascal_voc_annotation(xml_path)
#             all_ground_truths.append(ground_truths)
    
#     return all_ground_truths

# # Example usage
# annotation_dir = "./dataset/test/"  # Update this to the directory containing your XML files
# ground_truths = get_ground_truths_from_directory(annotation_dir)

# # Print ground truths for the first image


# mapped_ground_truth = [
#     [
#         {
#             "bbox": item["bbox"],
#             "label": item["label"],
#             "class": CLASSES.index(item["label"])
#         }
#         for item in sublist
#     ]
#     for sublist in ground_truths
# ]

# test_ground_truths = [
#   [{"bbox": [8, 29, 48, 188], "label": "chilli"}],

# ]

# sample = [
#   {
#     'boxes': tensor([[ 74.4506, 117.9982, 400.6461, 430.6247],
#         [ 51.4281,  64.0308, 432.2385, 455.4966],
#         [ 65.5099, 106.5663, 413.3722, 434.6424],
#         [ 54.6173,  70.6974, 426.3994, 422.5687]]), 
#         'labels': tensor([17, 18, 34, 30]),
#          'scores': tensor([0.0817, 0.0723, 0.0651, 0.0543])}, 
#          {'boxes': tensor([[ 15.6081,  51.2345, 506.7803, 460.8302]]), 'labels': tensor([1]), 'scores': tensor([0.9909])}, {'boxes': tensor([[  3.4930,  22.1772, 501.1651, 486.0865],
#         [ 28.0845,  20.1149, 506.7681, 477.9543]]), 'labels': tensor([ 1, 23]), 'scores': tensor([0.9980, 0.0951])}, {'boxes': tensor([[ 11.0675,   3.3082, 507.3584, 505.6017]]), 'labels': tensor([1]), 'scores': tensor([0.9967])}, {'boxes': tensor([[4.5879e-02, 6.4995e+00, 5.1007e+02, 5.1200e+02]]), 'labels': tensor([1]), 'scores': tensor([0.9949])}, {'boxes': tensor([[6.0843e+00, 7.0191e+00, 5.1200e+02, 5.0342e+02],
#         [1.4093e+01, 0.0000e+00, 5.1200e+02, 5.1200e+02],
#         [2.5557e+01, 6.7729e+00, 4.9846e+02, 4.9411e+02],
#         [0.0000e+00, 3.6934e-01, 5.1200e+02, 4.9919e+02]]), 'labels': tensor([ 1, 31, 23, 25]), 'scores': tensor([0.9854, 0.0777, 0.0756, 0.0725])}, {'boxes': tensor([[ 11.6003,   9.0338, 502.5264, 502.4651],
#         [ 14.8061,   0.0000, 510.6555, 495.2407],
#         [  0.0000,   0.7790, 512.0000, 512.0000]]), 'labels': tensor([ 1, 23, 28]), 'scores': tensor([0.9834, 0.1705, 0.1340])}, {'boxes': tensor([[  9.8996,  16.4349, 508.0935, 485.9194],
#         [  0.0000,  14.4564, 509.3543, 487.1161]]), 'labels': tensor([ 1, 23]), 'scores': tensor([0.9813, 0.1571])}] 


# print(mapped_ground_truth[0])


import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def evaluate_model(predictions, ground_truths, iou_threshold=0.5):
    """
    Evaluate a model's predictions to compute mAP and AR.
    Args:
        predictions: List of dictionaries, each with 'boxes', 'scores', and 'labels'.
        ground_truths: List of dictionaries, each with 'boxes' and 'labels'.
        iou_threshold: IoU threshold for considering a detection as TP.
    Returns:
        mean_ap: Mean Average Precision (mAP).
        average_recall: Average Recall (AR).
    """
    all_precisions = []
    all_recalls = []
    all_aps = []
    
    for class_id in range(1, num_classes + 1):  # Assume class 0 is background
        tp = 0
        fp = 0
        fn = 0
        precisions = []
        recalls = []
        
        for preds, gts in zip(predictions, ground_truths):
            pred_boxes = [box for box, label in zip(preds['boxes'], preds['labels']) if label == class_id]
            gt_boxes = [box for box, label in zip(gts['boxes'], gts['labels']) if label == class_id]
            
            matched_gt = set()
            for pred_box in pred_boxes:
                ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
                best_iou = max(ious) if ious else 0
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(gt_boxes[ious.index(best_iou)])
                else:
                    fp += 1
            
            fn += len(gt_boxes) - len(matched_gt)
        
        # Compute precision and recall for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        
        # Compute AP (average precision)
        ap = np.mean(precisions) if precisions else 0
        all_aps.append(ap)
    
    # Compute mAP and AR
    mean_ap = np.mean(all_aps)
    average_recall = np.mean(recalls)
    
    return mean_ap, average_recall

# Example usage
predictions = [
    {'boxes': [[50, 50, 150, 150]], 'scores': [0.9], 'labels': [1]},  # Predicted boxes
]
ground_truths = [
    {'boxes': [[48, 48, 152, 152]], 'labels': [1]},  # Ground-truth boxes
]

num_classes = 1  # Set the number of classes in your dataset
mAP, AR = evaluate_model(predictions, ground_truths)
print(f"mAP: {mAP:.4f}, AR: {AR:.4f}")

