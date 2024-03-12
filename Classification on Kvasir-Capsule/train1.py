#import os
#import time
import torch
import torchvision
import torch.nn as nn
#import torch.nn.functional as F
from torchvision.transforms import v2
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
from tqdm.auto import tqdm
from dataset import kvasir_cap
from model import resnet_X
from utils import save_checkpoint, load_checkpoint

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
#RANDOM_SEED = 42
DATASET_PATH = "D:\\kvasir_CAPSULE\\data\\labelled_images"
MODEL_SAVE_PATH = "D:\\kvasir_CAPSULE\\classification_implementation\\model_resnet50.pth"
LEARNING_RATE = 0.01
CLASSES = 14
EPOCH = 5


# Define augmentations and transformations

#transforms = v2.Compose([
#    v2.ToImage(),
#    v2.ToDtype(torch.float32, scale=True),
#    v2.RandomHorizontalFlip(p = 0.5),
#    v2.RandomRotation(degrees=(0, 180)),
#])

data_transforms = v2.Compose([
    #v2.Grayscale(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

# Data loaders
train_dl, val_dl, test_dl = kvasir_cap(path=DATASET_PATH,
                                  batch_size = BATCH_SIZE,
                                  transforms=data_transforms,
                                  )


# Model
## Self Defined
resnet_18 = resnet_X(layers=[2, 2, 2, 2],       #for resnet18
                     classes=CLASSES).to(DEVICE)

## From torchvision with pre-trained ImageNet weights
model_resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
#model_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)       # 1 as greyscale
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=CLASSES)
model_resnet.to(DEVICE)


model_densnet121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
#model_densnet121.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)       # 1 as greyscale
num_ftrs = model_densnet121.classifier.in_features
model_densnet121.classifier = nn.Linear(in_features=num_ftrs, out_features=CLASSES)
model_densnet121.to(DEVICE)


# Train
metric = MulticlassF1Score(num_classes=CLASSES).cuda()
prec = MulticlassPrecision(average='macro', num_classes=CLASSES).cuda()
recall = MulticlassRecall(average='macro', num_classes=CLASSES).cuda()
acc = MulticlassAccuracy(average='macro', num_classes=CLASSES).cuda()
conf_matrix = MulticlassConfusionMatrix(num_classes=CLASSES).cuda()     # Avoid macro-averaging within micro-average



def train(epochs, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    for epoch in range(epochs):
        running_loss = 0
        for batch_idx, (data, targets) in enumerate(tqdm(train_dl)):
            data, targets = data.cuda(), targets.cuda()

            #forward
            scores = model(data)
            loss = criterion(scores, targets)
            running_loss += loss.item()
            
            #backward
            optimizer.zero_grad()
            loss.backward()

            #gradient descent or adam step
            optimizer.step()

        avg_loss = running_loss/len(train_dl)
        print(f"| Epoch {epoch+1}/{epochs} running loss: {running_loss} average loss: {avg_loss}")
        check_accuracy(val_dl, model)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename = MODEL_SAVE_PATH)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            scores = model(x)

            _, predictions = torch.max(scores, 1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} corrent with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()



if __name__ == "__main__":
    train(epochs=EPOCH, model = model_resnet)
    model = model_resnet
    model.to(DEVICE)
    load_checkpoint(torch.load(MODEL_SAVE_PATH), model)
    model.eval()
    check_accuracy(test_dl, model)