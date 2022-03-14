import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1
BATCH_SIZE = 16
PIN_MEMORY = True
LOAD_MODEL = True
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

#define size to which images are to be resized
IMAGE_SIZE = [224, 224] # can be changed depending on the dataset

train_transforms = A.Compose([
    A.Resize(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1],),
    A.RandomCrop(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1]),
    A.Rotate(20),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
    #    max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1],),
    A.Normalize(
       mean=[0, 0, 0],
       std=[1, 1, 1],
       #max_pixel_value=255.0,
    ),
    ToTensorV2(),
])
