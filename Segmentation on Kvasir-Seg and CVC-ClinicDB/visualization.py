import os
import random
import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import v2
from utils import *
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from models import UNET
import imp

DATA_PATH = "D:\\kvasir-seg\\Kvasir-SEG\\Data"
MODEL_PATH1 = "D:\\FINAL\\vanilla_UNet.pth"
MODEL_PATH2 = "D:\\FINAL\\SMP_UNet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#MODEL = smp.Unet(
#    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#    encoder_weights="imagenet",     # use imagenet pre-trained weights for encoder initialization
#    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#    classes=1,                      # model output channels (number of classes in your dataset)
#).to(DEVICE)
#load_checkpoint(torch.load(MODEL_PATH), MODEL)
#MODEL.eval()

def main():
    st.title("Segmentation Model Inference")

    # Model selection
    model_option = st.selectbox("Select Model", ["Vanilla", "ImageNet Pretrained"]) # Add your model names here

    # Load model based on selection
    if model_option == "Vanilla":
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        load_checkpoint(torch.load(MODEL_PATH1), model)
        model.eval()
    elif model_option == "ImageNet Pretrained":
        model = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use imagenet pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        ).to(DEVICE)
        load_checkpoint(torch.load(MODEL_PATH2), model)
        model.eval()
    #else:
    #    model_path = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\UNet_ResNet50BackEnd_ImageNetWeights_FocalLoss_"+"f31.pth"


    number = int(st.number_input('How many images do you want to test on? ', format="%f"))
    _, _, test_dl = get_loaders(DATA_PATH, batch_size=number)

    for batch_idx, (data, target) in enumerate(test_dl):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        with torch.inference_mode():
            preds = torch.sigmoid(model(data))
        preds = torch.round(preds).float()
        break
    
    images = data.permute(0, 3, 2, 1).cpu().numpy()
    original_masks = target.permute(0, 3, 2, 1).squeeze(3).cpu().numpy()
    predicted_masks = preds.permute(0, 3, 2, 1).squeeze(3).cpu().numpy()

    for i in range(images.shape[0]):
        
        st.subheader(f"Image: {i+1}")
        fig, axes = plt.subplots(1, 3, figsize=(25, 5))
        # Display original image
        axes[0].imshow(images[i])
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        # Display original mask
        axes[1].imshow(original_masks[i])
        axes[1].set_title("Original Mask")
        axes[1].axis('off')
        # Display predicted mask
        axes[2].imshow(predicted_masks[i]) 
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')
        st.pyplot(fig)


if __name__ == "__main__":
    main()