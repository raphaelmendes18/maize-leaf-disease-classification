__author__ = 'Raphael Mendes'
__date__ = '06/22/21'

import pandas as pd
import torch
from neural_network.modified_letnet import ModifiedLeNet
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import MaizePlantVillageDataset
from training.utils import training_loop, get_accuracy, get_classification_report
import argparse

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--PREPROCESS_DIR', type=str)
parser.add_argument('--N_EPOCHS', type=int)
parser.add_argument('--SAVE_DIR', type=float)

# PARAMETERS
# BASE_DIR = '/content/drive/MyDrive/PlantVillage/color'
# OUTPUT_FILE = '/content/drive/MyDrive/PlantVillage/color/dataset_reference.csv'
# TRAIN_SIZE = 0.5
# TEST_SIZE = 0.5
# EXP_NAME = 'exp_1'
# PREPROCESS_BASE_DIR = f'/content/drive/MyDrive/PlantVillage/preprocessing/PCA/{EXP_NAME}'

args = parser.parse_args()
# '/content/drive/MyDrive/PlantVillage/preprocessing/PCA/exp_1'
PREPROCESS_DIR = args.PREPROCESS_DIR
N_EPOCHS = args.N_EPOCHS  # 1000
# '/content/drive/MyDrive/PlantVillage/preprocessing/PCA/exp_1/model_1000.pth'
SAVE_DIR = args.SAVE_DIR

# Singleton
BATCH_SIZE = 64
IMG_SIZE = 64
N_CLASSES = 4
RANDOM_SEED = 42
LEARNING_RATE = 0.001

# Class Keys
key_to_class = {0: 'CR', 1: 'H', 2: 'NLB', 3: 'GLP'}

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device is {DEVICE}.')

torch.manual_seed(RANDOM_SEED)

model = ModifiedLeNet(N_CLASSES).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# define transforms
transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor()])

# create datasets
plant_dataset_train = MaizePlantVillageDataset(
    csv_file=PREPROCESS_DIR+'/train.csv', transform=transform)
plant_dataset_val = MaizePlantVillageDataset(
    csv_file=PREPROCESS_DIR+'/val.csv', transform=transform)
plant_dataset_test = MaizePlantVillageDataset(
    csv_file=PREPROCESS_DIR+'/test.csv', transform=transforms.Compose([transforms.ToTensor()]))

# # define the data loaders
train_loader = DataLoader(dataset=plant_dataset_train,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=plant_dataset_val,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

# # Train Model
model, optimizer, _ = training_loop(
    model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

# # Store Model
torch.save(model.state_dict(), SAVE_DIR)

# # Test Model

# Load model for testing
model = ModifiedLeNet(N_CLASSES)
model.load_state_dict(torch.load(SAVE_DIR))
model.eval()
model = model.to(DEVICE)

# Data Loader for Test Set
test_loader = DataLoader(dataset=plant_dataset_test,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

test_acc = get_accuracy(model, test_loader, device=DEVICE)
print(f'Test accuracy: {100 * test_acc:.2f}\t')

# # Print other metrics
get_classification_report(model, test_loader, device=DEVICE)
