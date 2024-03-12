
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = 14


def check_accuracy(loader, model):
    criterion = nn.CrossEntropyLoss()
    num_correct = 0
    num_samples = 0
    loss = 0
    metric1 = MulticlassF1Score(average=None, num_classes=CLASSES).cuda()
    prec1 = MulticlassPrecision(average=None, num_classes=CLASSES).cuda()
    recall1 = MulticlassRecall(average=None, num_classes=CLASSES).cuda()
    acc1 = MulticlassAccuracy(average=None, num_classes=CLASSES).cuda() 
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            scores = model(x)
            _, predictions = torch.max(scores, 1)
            loss += criterion(scores, y).item()
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            metric1.update(predictions, y)
            prec1.update(predictions, y)
            recall1.update(predictions, y)
            acc1.update(predictions, y)
        avg_loss = loss/len(loader)
        print(f"Got {num_correct}/{num_samples} corrent with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
        print(f"Got {num_correct}/{num_samples} corrent with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
        print(f"| Accuracy: {acc1.compute():.3f}, precision: {prec1.compute():.3f}, recall: {recall1.compute():.3f}, F1Score: {metric1.compute():.3f}, average loss: {avg_loss:.3f}")

    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])