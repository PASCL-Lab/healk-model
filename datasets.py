import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform, diagnose_image_loading
# the dataset class


class GroceryDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(
            f"{self.dir_path}/*.[jp]*g") + glob.glob(f"{self.dir_path}/*.png") + glob.glob(f"{self.dir_path}/*.webp")
        self.all_images = [image_path.split(
            os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format

        if image is None:
            print(image_path, image, "HELP IMAGE NOT LOADED")
            diagnose_image_loading(image_path)
            return
            # raise ValueError(f"Image at path {image_path} not loaded")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4].rstrip('.') + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        if annot_file_path.endswith("..xml"):
            print(annot_file_path)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list


            labels.append(self.classes.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        # bounding box to tensor
        
        
        if len(boxes) == 0:
            print(image_path)
            boxes = torch.zeros(0, 4, dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if boxes.dim() == 1:
            # Single bounding box case
            boxes = boxes.unsqueeze(0)

        if boxes.dim() == 2:
            # Proceed with existing 2D tensor logic
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # area of the bounding boxes
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # print(f"EXISTING", labels)
        # labels to tensor
        torch_labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch_labels
        # target["class_labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # target["image_name"] = image_name
        # apply the image transforms

        # print("TARGET LENGTH", target.length)
        try:
            # return sample

            if self.transforms:
                sample = self.transforms(image=image_resized,
                                         bboxes=target['boxes'],
                                         labels=labels)

                image_resized = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])
                # print(f"Image: {image_name},")
                # print(f"Boxes shape: {target['boxes'].shape}")
        except Exception as e:
            print(f"Exception: {e}")

        return image_resized, target

    def __len__(self):
        return len(self.all_images)
# prepare the final datasets and data loaders


def create_train_dataset():
    train_dataset = GroceryDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    return train_dataset


def create_valid_dataset():
    valid_dataset = GroceryDataset(
        VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    return valid_dataset


def create_train_loader(train_dataset, num_workers=1):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=1):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    print("STARTING SANITY CHECK")

    dataset = GroceryDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = len(dataset)
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        response = dataset[i]
        # visualize_sample(image, target)

    print("ALL SUCCESS")

