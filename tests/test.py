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


CLASSES = [
    # 'background',
    "almond",
    "apple",
    "avocado",
    "banana",
    "beetroot",
    "bell pepper",
    "blueberries",
    "broccoli",
    "brussels",
    "cabbage",
    "carrot",
    "cauliflower",
    "chilli",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "parsley",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soybeans",
    "spinach",
    "strawberries",
    "tomato",
    "turnip",
    "walnut",
    "watermelon"
]
num_classes = len(CLASSES)

# Assuming you have a dataset class

# Define transforms for the dataset
# transform = get_valid_transform()

# Load dataset and create DataLoader
# # train_dataset = create_train_dataset()
# valid_dataset = create_valid_dataset()
# # train_loader = create_train_loader(train_dataset, NUM_WORKERS)
# test_loader = create_valid_loader(valid_dataset, NUM_WORKERS)

# # test_dataset = CustomDataset(transforms=transform)
# # test_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# # Load your model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
#     weights=None
# )

# # Get the number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features

# # Define a new head for the detector with the required number of classes
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# model.load_state_dict(torch.load('outputs/best_model.pth')['model_state_dict'])
# model.eval()

# # Instantiate the mAP metric
# map_metric = MeanAveragePrecision()

# print(test_loader, "TEST LOADER LENGTH")


# prog_bar = tqdm(test_loader, total=len(test_loader))


# for i, data in enumerate(prog_bar):
#     print("HELLOOO WODTLD")
#     images, targets = data

#     images = list(image.to(DEVICE) for image in images)
#     targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]


#     # Move data to device if using GPU
#     # images = [img.to(DEVICE) for img in images]
#     # targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

#     # with torch.no_grad():
#     #     # Get predictions from the model
#     #     predictions = model(images, targets)

#     # # Format predictions for the mAP metric
#     # formatted_predictions = [
#     #     {
#     #         "boxes": pred["boxes"].cpu(),
#     #         "scores": pred["scores"].cpu(),
#     #         "labels": pred["labels"].cpu(),
#     #     }
#     #     for pred in predictions
#     # ]

#     # # Format targets for the mAP metric
#     # formatted_targets = [
#     #     {
#     #         "boxes": target["boxes"].cpu(),
#     #         "labels": target["labels"].cpu(),
#     #     }
#     #     for target in targets
#     # ]

#     # Update the mAP metric
#     # map_metric.update(formatted_predictions, formatted_targets)

# # Compute the final results
# # results = map_metric.compute()
# # print(f"Mean Average Precision (mAP): {results['map']:.4f}")
# # print(f"Average Recall: {results['mar_100']:.4f}")

# train_dataset = create_train_dataset()

# map_metric = MeanAveragePrecision()


def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    # prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for data in valid_data_loader:
        images, targets = data

        print(len(images), targets, "FIRST IT")

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        print(loss_dict, "THE DICT", len(images))
        # losses = sum(loss for loss in loss_dict.values())
        # loss_value = losses.item()
        # val_loss_list.append(loss_value)
        # val_loss_hist.send(loss_value)
        # val_itr += 1
        # # update the loss value beside the progress bar for each iteration
        # prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


valid_dataset = create_valid_dataset()
# train_loader = create_train_loader(train_dataset, NUM_WORKERS)
valid_loader = create_valid_loader(valid_dataset, 0)
# print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")
# initialize the model and move to the computation device
model = create_model(num_classes=num_classes)
model.load_state_dict(torch.load('outputs/best_model.pth')['model_state_dict'])
model.eval()

model = model.to(DEVICE)
# get the model parameters
params = [p for p in model.parameters() if p.requires_grad]
# define the optimizer
optimizer = torch.optim.SGD(
    params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# initialize the Averager class
# train_loss_hist = Averager()
# val_loss_hist = Averager()
train_itr = 1
val_itr = 1
# train and validation loss lists to store loss values of all...
# ... iterations till ena and plot graphs for all iterations
train_loss_list = []
val_loss_list = []
# name to save the trained model with
MODEL_NAME = 'model'
# whether to show transformed images from data loader or not
# if VISUALIZE_TRANSFORMED_IMAGES:
#     from utils import show_tranformed_image
#     show_tranformed_image(train_loader)
# initialize SaveBestModel class
# save_best_model = SaveBestModel()
# print("ABOUT TO START TRAINING")
# start the training epochs
val_loss = validate(valid_loader, model)
