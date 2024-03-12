import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models import UNET
import segmentation_models_pytorch as smp
from utils import (
    #load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 25
#LOAD_MODEL = False
DATA_PATH = "D:\\kvasir-seg\\Kvasir-SEG\\Data"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)                                             # for transforms.v2
        #targets = targets.to(device=DEVICE).float().unsqueeze(1)                       # for albumentations
        #print(f"\nInput image batch shape: {data.size()}, Input target batch shape: {targets.size()}")             # sanity check
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #print(f'\nData dim: {data.size()}, Target dim: {targets.size()}, Preds dim: {predictions.size()}')     # sanity check
            #print(f'\nData type: {data.dtype}, Target type: {targets.dtype}, Preds type: {predictions.dtype}')     # sanity check
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Total loss: {total_loss}, average loss: {avg_loss}.")


def main():
    '''
    From pytorch hub, with pretrained weights 
    '''
    #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True).to(DEVICE)       
    '''
    Self defined, no pretrained weights
    ''' 
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    '''
    From segmentation-models-pytorch
    '''
    #model = smp.Unet(
    #    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #    classes=1,                      # model output channels (number of classes in your dataset)
    #    ).to(DEVICE)
           
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader, test_loader = get_loaders(DATA_PATH)

    #if LOAD_MODEL:
    #    load_checkpoint(torch.load("my_checkpoint.pth"), model)


    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f'| Epoch {epoch}:')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }
        
        save_checkpoint(checkpoint, 'vanilla_UNet.pth')

        # check accuracy
        print("\nValidation:")
        check_accuracy(val_loader, model, device=DEVICE)

    print("\nTest:")
    check_accuracy(test_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()