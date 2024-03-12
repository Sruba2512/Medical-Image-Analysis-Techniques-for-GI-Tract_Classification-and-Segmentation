import os
import torch
from torchvision.transforms import v2
#import albumentations as A
#from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from PIL import Image
from models import UNET
import segmentation_models_pytorch as smp
from utils import load_checkpoint, get_loaders, check_accuracy

TEST_IMAGES_FOLDER = "D:\\kvasir-seg\\Kvasir-SEG\\implementation\\test\\img"
OUTPUT_MASKS_FOLDER = "D:\\kvasir-seg\\Kvasir-SEG\\implementation\\test\\mask"
DATA_PATH = "D:\\kvasir-seg\\Kvasir-SEG\\Data"
MODEL_PATH = "D:\\kvasir-seg\\Kvasir-SEG\\implementation\\my_checkpoint.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load(MODEL_PATH), MODEL)
MODEL.eval()


#MODEL = smp.Unet(
#    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#    classes=1,                      # model output channels (number of classes in your dataset)
#).to(DEVICE)
#load_checkpoint(torch.load(MODEL_PATH), MODEL)
#MODEL.eval()

trans = v2.Compose([
    v2.Resize((336, 336)),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
    ])

#trans = A.Compose([
#    A.Resize(352, 352),
#    #A.HorizontalFlip(p=0.5),
#    #A.VerticalFlip(p=0.5),
#    A.ToFloat(),
#    ToTensorV2()
#    ])

_, _, test_loader = get_loaders(DATA_PATH)

def predict(input_path, output_path, model = MODEL):
    test_img_list = os.listdir(input_path)
    for img_name in test_img_list:
        img_path = os.path.join(input_path, img_name)
        img_tensor = trans(Image.open(img_path).convert('RGB'))
        img_tensor = img_tensor.unsqueeze(0)        # add a batch dimension

        with torch.no_grad():
            output = model(img_tensor.to(device=DEVICE))

        preds = torch.sigmoid(output)
        predicted_mask = (preds > 0.5).float().cpu()

        mask_array = predicted_mask.squeeze().numpy()
        mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))
        mask_image = v2.Resize((336, 336))(mask_image)
        mask_image.save(os.path.join(output_path, f'{img_name.split(".")[0]}_mask.png'))


def main():
    #predict(TEST_IMAGES_FOLDER, OUTPUT_MASKS_FOLDER)
    #print("| Prediction and saving complete.")
    check_accuracy(test_loader, MODEL)


if __name__ == '__main__':
    main()