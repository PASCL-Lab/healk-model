import subprocess

# Run another Python script
subprocess.run(['python', 'datasets.py'])
subprocess.run(['python', 'engine.py'])
a = {'boxes': tensor([[206.6032, 392.6349, 288.3810, 439.8730],
        [ 87.8730, 353.6931, 157.7143, 407.0265],
        [ 84.9524,  88.2116, 416.8889, 392.6349]]), 
        'labels': tensor([0, 0, 0]), 
        'class_labels': [0, 0, 0], 
        'area': tensor([  3863.0264,   3724.8682, 101049.2031]), 
        'iscrowd': tensor([0, 0, 0]), 'image_id': tensor([0])}


  b = tensor([[0.9616],
        [0.9245]], grad_fn=<DivBackward0>)


iou = tensor([[0.9616],
        [0.9245]], grad_fn=<DivBackward0>)

ace =detections = [
    {
        'boxes': [
            [74.4506, 117.9982, 400.6461, 430.6247],
            [51.4281, 64.0308, 432.2385, 455.4966],
            [65.5099, 106.5663, 413.3722, 434.6424],
            [54.6173, 70.6974, 426.3994, 422.5687],
        ],
        'labels': [17, 18, 34, 30],
        'scores': [0.0817, 0.0723, 0.0651, 0.0543],
    },
    {
        'boxes': [
            [15.6081, 51.2345, 506.7803, 460.8302],
        ],
        'labels': [1],
        'scores': [0.9909],
    },
    {
        'boxes': [
            [3.4930, 22.1772, 501.1651, 486.0865],
            [28.0845, 20.1149, 506.7681, 477.9543],
        ],
        'labels': [1, 23],
        'scores': [0.9980, 0.0951],
    },
    {
        'boxes': [
            [11.0675, 3.3082, 507.3584, 505.6017],
        ],
        'labels': [1],
        'scores': [0.9967],
    },
    {
        'boxes': [
            [0.045879, 6.4995, 510.07, 512.0],
        ],
        'labels': [1],
        'scores': [0.9949],
    },
    {
        'boxes': [
            [6.0843, 7.0191, 512.0, 503.42],
            [14.093, 0.0, 512.0, 512.0],
            [25.557, 6.7729, 498.46, 494.11],
            [0.0, 0.36934, 512.0, 499.19],
        ],
        'labels': [1, 31, 23, 25],
        'scores': [0.9854, 0.0777, 0.0756, 0.0725],
    },
    {
        'boxes': [
            [11.6003, 9.0338, 502.5264, 502.4651],
            [14.8061, 0.0, 510.6555, 495.2407],
            [0.0, 0.7790, 512.0, 512.0],
        ],
        'labels': [1, 23, 28],
        'scores': [0.9834, 0.1705, 0.1340],
    },
    {
        'boxes': [
            [9.8996, 16.4349, 508.0935, 485.9194],
            [0.0, 14.4564, 509.3543, 487.1161],
        ],
        'labels': [1, 23],
        'scores': [0.9813, 0.1571],
    },
]


label =detections = [
    {
        'boxes': [
            [206.6032, 392.6349, 288.3810, 439.8730],
            [87.8730, 353.6931, 157.7143, 407.0265],
            [84.9524, 88.2116, 416.8889, 392.6349],
        ],
        'labels': [0, 0, 0],
        'class_labels': [0, 0, 0],
        'area': [3863.0264, 3724.8682, 101049.2031],
        'iscrowd': [0, 0, 0],
        'image_id': 0,
        'image_name': 'almond_10.webp',
    },
    {
        'boxes': [
            [5.1200, 46.0800, 512.0000, 460.8000],
        ],
        'labels': [1],
        'class_labels': [1],
        'area': [210213.2656],
        'iscrowd': [0],
        'image_id': 1,
        'image_name': 'apple_1182.jpg',
    },
    {
        'boxes': [
            [5.1200, 5.1200, 512.0000, 486.4000],
        ],
        'labels': [1],
        'class_labels': [1],
        'area': [243951.2031],
        'iscrowd': [0],
        'image_id': 2,
        'image_name': 'apple_1313.jpg',
    },
    {
        'boxes': [
            [5.1200, 5.1200, 512.0000, 496.6400],
        ],
        'labels': [1],
        'class_labels': [1],
        'area': [249141.6719],
        'iscrowd': [0],
        'image_id': 3,
        'image_name': 'apple_173.jpg',
    },
    {
        'boxes': [
            [1.1558, 1.2104, 512.0000, 512.0000],
        ],
        'labels': [1],
        'class_labels': [1],
        'area': [260933.9375],
        'iscrowd': [0],
        'image_id': 4,
        'image_name': 'apple_1750.jpg',
    },
    {
        'boxes': [
            [5.1200, 5.1200, 512.0000, 501.7600],
        ],
        'labels': [1],
        'class_labels': [1],
        'area': [251736.8906],
        'iscrowd': [0],
        'image_id': 5,
        'image_name': 'apple_41.jpg',
    },
    {
        'boxes': [
            [5.1200, 5.1200, 512.0000, 512.0000],
        ],
        'labels': [1],
        'class_labels': [1],
        'area': [256927.3438],
        'iscrowd': [0],
        'image_id': 6,
        'image_name': 'apple_418.jpg',
    },
    {
        'boxes': [
            [5.1200, 20.4800, 512.0000, 491.5200],
        ],
        'labels': [1],
        'class_labels': [1],
        'area': [238760.7500],
        'iscrowd': [0],
        'image_id': 7,
        'image_name': 'apple_427.jpg',
    },
]


# Iterate based on the length of true_labels
for idx in range(len(true_labels)):
    # Check if the predicted label corresponds to the current true label
    if pred_labels[idx].item() == true_labels[idx].item():
        # Compute IoU between all predicted boxes and true boxes
        iou = box_iou(pred_boxes[idx].unsqueeze(0), true_boxes[idx].unsqueeze(0))

        # Only consider predictions with IoU > 0.5
        if iou.item() > 0.5:
            all_preds.append(pred_labels[idx].item())
            all_labels.append(true_labels[idx].item())


# Ensure we don't exceed the shorter length
# min_len = min(len(pred_labels), len(true_labels))

# for idx in range(min_len):
#     if pred_labels[idx].item() == true_labels[idx].item():
#         iou = box_iou(pred_boxes[idx].unsqueeze(0), true_boxes[idx].unsqueeze(0))
#         if iou.item() > 0.5:
#             all_preds.append(pred_labels[idx].item())
#             all_labels.append(true_labels[idx].item())


 preds = [
...   dict(
...     boxes=tensor([[258.0, 41.0, 606.0, 285.0]]),
...     scores=tensor([0.536]),
...     labels=tensor([0]),
...   )
... ]
>>> target = [
...   dict(
...     boxes=tensor([[214.0, 41.0, 562.0, 285.0]]),
...     labels=tensor([0]),
...   )
... ]

[{'boxes': tensor([[ 74.4506, 117.9982, 400.6461, 430.6247],
        [ 51.4281,  64.0308, 432.2385, 455.4966],
        [ 65.5099, 106.5663, 413.3722, 434.6424],
        [ 54.6173,  70.6974, 426.3994, 422.5687]]), 
        'labels': tensor([17, 18, 34, 30]), 
        'scores': tensor([0.0817, 0.0723, 0.0651, 0.0543])}
  ]
  

  ({'boxes': tensor([[206.6032, 392.6349, 288.3810, 439.8730],
        [ 87.8730, 353.6931, 157.7143, 407.0265]]), 'labels': tensor([0, 0]), 'class_labels': [0, 0], 'area': tensor([3863.0264, 3724.8682]), 'iscrowd': tensor([0, 0]), 'image_id': tensor([0]), 'image_name': 'almond_10.webp'},)


{'boxes': tensor([[ 15.6081,  51.2345, 506.7803, 460.8302]]), 'labels': tensor([1]), 'scores': tensor([0.9909])} 
{'boxes': tensor([[  5.1200,  46.0800, 512.0000, 460.8000]]), 'labels': tensor([1]), 'class_labels': [1], 'area': tensor([210213.2656]), 'iscrowd': tensor([0]), 'image_id': tensor([0]), 'image_name': 'apple_1182.jpg'}



import torch

# Example predictions and target
predictions = {'boxes': torch.tensor([[15.6081, 51.2345, 506.7803, 460.8302]]), 
               'labels': torch.tensor([1]), 
               'scores': torch.tensor([0.9909])}

target = {'boxes': torch.tensor([[5.1200, 46.0800, 512.0000, 460.8000]]), 
          'labels': torch.tensor([1]), 
          'class_labels': [1], 
          'area': torch.tensor([210213.2656]), 
          'iscrowd': torch.tensor([0]), 
          'image_id': torch.tensor([0]), 
          'image_name': 'apple_1182.jpg'}

# Calculate accuracy
def calculate_accuracy(predictions, target):
    pred_labels = predictions['labels']
    target_labels = target['labels']
    
    correct = (pred_labels == target_labels).sum().item()  # Number of correct predictions
    total = target_labels.size(0)  # Total number of ground truth labels
    
    accuracy = correct / total if total > 0 else 0  # Handle case where no targets are present
    return accuracy

# Call the function and print accuracy
accuracy = calculate_accuracy(predictions, target)
print(f'Accuracy: {accuracy:.4f}')
