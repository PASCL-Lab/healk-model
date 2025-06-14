import torch
BATCH_SIZE = 8
RESIZE_TO = 512
NUM_EPOCHS = 20
NUM_WORKERS = 1
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
# TRAIN_DIR = '/home/iyanu/projects/def-iadaji/iyanu/model/dataset/train/'
TRAIN_DIR = './dataset/train/'
# validation images and XML files directory
# VALID_DIR = '/home/iyanu/projects/def-iadaji/iyanu/model/dataset/test/'
VALID_DIR = './dataset/test/'

# classes: 0 index is reserved for background
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
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

OUT_DIR = 'outputs'
SAVE_PLOTS_EPOCH = 2  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2  # save model after these many epochs
