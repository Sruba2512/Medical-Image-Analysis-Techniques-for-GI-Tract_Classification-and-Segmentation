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
from utils import save_checkpoint, load_checkpoint, check_accuracy


# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
#RANDOM_SEED = 42
DATASET_PATH = "D:\\kvasir_CAPSULE\\data\\labelled_images"
#MODEL_SAVE_PATH = "C:\\Users\\sr1ja\\Desktop\New folder (2)\\New folder\\saved_models\\model_resnet18.pth"
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
#metric = MulticlassF1Score(num_classes=CLASSES).cuda()
#prec = MulticlassPrecision(average='macro', num_classes=CLASSES).cuda()
#recall = MulticlassRecall(average='macro', num_classes=CLASSES).cuda()
#acc = MulticlassAccuracy(average='macro', num_classes=CLASSES).cuda()
#conf_matrix = MulticlassConfusionMatrix(num_classes=CLASSES).cuda()     # Avoid macro-averaging within micro-average



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
        print(f"| Epoch {epoch+1}/{epochs} total loss: {running_loss}, average loss: {avg_loss}")
        model.eval()
        with torch.inference_mode():
            num_correct = 0
            num_samples = 0
            val_loss = 0
            metric = MulticlassF1Score(average=None, num_classes=CLASSES).cuda()
            prec = MulticlassPrecision(average=None, num_classes=CLASSES).cuda()
            recall = MulticlassRecall(average=None, num_classes=CLASSES).cuda()
            acc = MulticlassAccuracy(average=None, num_classes=CLASSES).cuda()
            for inputs, labels in val_dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                scores  = model(inputs)
                _, predictions = torch.max(scores, 1)
                loss = criterion(scores, labels)
                val_loss += loss.item()
                num_correct += (predictions == labels).sum()
                num_samples += predictions.size(0)
                metric.update(predictions, labels)
                prec.update(predictions, labels)
                recall.update(predictions, labels)
                acc.update(predictions, labels)
            avg_val_loss = val_loss/len(val_dl)
            print(f"Got {num_correct}/{num_samples} corrent with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
            print(f"| Total validation loss: {val_loss:.3f}, average validation loss: {avg_val_loss}, accuracy: {acc.compute():.3f}, precision: {prec.compute():.3f}, recall: {recall.compute():.3f}, F1Score: {metric.compute():.3f}")
        model.train()
    print('Saving model...')
    checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }
    save_checkpoint(checkpoint, MODEL_SAVE_PATH)
    print(f'Model saved at {MODEL_SAVE_PATH}')




if __name__ == "__main__":
    train(epochs=EPOCH, model = model_resnet)
    print("\nOn test data: ")
    load_checkpoint(torch.load(MODEL_SAVE_PATH), model_resnet)
    model_resnet.eval()
    check_accuracy(test_dl, model_resnet)

#def train(epochs, model):
#    optimizer = torch.optim.Adam(model.parameters())
#    loss_func = nn.CrossEntropyLoss()
#    for epoch in range(epochs):
#        #tinme.sleep(1)
#        model.train()
#        running_loss = 0
#        for _, (inputs, labels) in enumerate(tqdm(train_dl)):
#            inputs, labels = inputs.cuda(), labels.cuda()
#            optimizer.zero_grad()
#            with torch.set_grad_enabled(True):
#                logits, outputs = torch.max(model(inputs), dim=1)
#                #print('\nLogits:',logits)
#                #print('\nOutputs:',outputs)
#                #print('\nLabels:',labels)
#                # preds = torch.round(outputs.sigmoid()).squeeze()
#                loss = loss_func(logits, labels.float())
#                #print('\n',loss)
#                running_loss += loss.item()
#                loss.backward()
#                optimizer.step()
#        print(f"| Epoch {epoch+1}/{epochs} running loss: {running_loss}")
#        model.eval()
#        with torch.inference_mode():
#            valid_loss = 0
#            for inputs, labels in val_dl:
#                inputs, labels = inputs.cuda(), labels.cuda()
#                logits, preds = torch.max(model(inputs), dim=1)
#                loss = loss_func(logits, labels.float())
#                valid_loss += loss.item()
#                metric.update(preds, labels)
#                prec.update(preds, labels)
#                recall.update(preds, labels)
#                acc.update(preds, labels)
#                conf_matrix.update(preds, labels)
#            print(f"|Total validation loss for epoch {epoch+1}: {valid_loss:.3f},\naccuracy: {acc.compute():.3f},\nprecision: {prec.compute():.3f},\nrecall: {recall.compute():.3f},\nF1Score: {metric.compute():.3f},\nConfusion Matrix: {conf_matrix.compute()}")
#    print('Saving model...')
#    torch.save(model.state_dict(), MODEL_SAVE_PATH)
#    print(f'Model saved at {MODEL_SAVE_PATH}')
#

#def main():
#    # Defining the transformations 
#    data_transforms = v2.Compose([
#    v2.ToImage(),
#    v2.ToDtype(torch.float32, scale=True),
#    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#    ])
#
#    # Defining the model
#    model_resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
#    num_ftrs = model_resnet.fc.in_features
#    model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=1)
#    model_resnet.to(DEVICE)
#    
#    # Data loaders
#    train_dl, val_dl, _ = BleedLoader(path=DATASET_PATH, 
#                                      batch_size = BATCH_SIZE, 
#                                      transforms=data_transforms)
#    
#    
#    pass
