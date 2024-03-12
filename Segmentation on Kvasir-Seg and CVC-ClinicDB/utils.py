# To import the PyTorch library
import torch

# To allow importing specific image transformations
from torchvision.transforms import v2

from dataset import KvasirSeg

# To define a function for generating data loaders for training and validation
def get_loaders(path, splits = [0.7, 0.15], batch_size = 8):

    #random_seed = 42
    #torch.manual_seed(random_seed)

    # To define image transformation pipeline
    trans = v2.Compose([
        v2.Resize((352, 352)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True)
    ]) 

    ## Dataset creation and `DataLoader` initialization for training and validation
    dataset = KvasirSeg(root_dir = path, transform = trans)
    train_len = int(splits[0] * len(dataset))
    val_len = int(splits[1] * len(dataset))
    test_len = len(dataset) - (train_len + val_len)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size = batch_size)

    return train_dl, val_dl, test_dl


# To define a function to save training checkpoints
def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# To define a function for loading checkpoints into the model
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    IoU = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            intersection = (preds * y).sum()
            union = (preds + y).sum() - intersection
            IoU += (intersection + 1e-8) / (union + 1e-8)

    print(f"\nPredicted {num_correct} pixels correct out of {num_pixels} pixels, with accuracy: {num_correct/num_pixels*100:.2f}%")
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"IoU: {IoU / len(loader)}")
    model.train()





