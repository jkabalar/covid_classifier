import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import torch
import tqdm
import config
from dataset import ImageFolder

#Calculate Mean and STD for this dataset
all_ds = ImageFolder(root_dir="dataset/", transform=config.val_transforms)

train_loader = DataLoader(all_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,pin_memory=config.PIN_MEMORY, shuffle=True)
# placeholders
psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs, label in tqdm.tqdm(train_loader):
    psum    += inputs.sum(axis        = [0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

####### FINAL CALCULATIONS

# pixel count
count = len(train_loader) * config.IMAGE_SIZE[0] * config.IMAGE_SIZE[1]

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))
# The mean and std of this dataset are [0.4954, 0.4957, 0.4963] and std = [0.2287, 0.2287, 0.2289].