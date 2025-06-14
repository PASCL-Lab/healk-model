import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


WEIGHTS_PATH = "./outputs/best_model.pth"


def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None)

    # state_dict = torch.load(WEIGHTS_PATH)
    # model.load_state_dict(state_dict)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
