# To import the 'os' module thereby enabling interactions with the Operating System
import os

# To import the 'Image' module thereby providing extensive image processing functionalities
from PIL import Image

# To import 'Dataset' module for creating custom datasets in PyTorch
from torch.utils.data import Dataset

class KvasirSeg(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform

        # To initialize attributes as empty lists
        self.image_paths = []
        self.mask_paths = []

        # To construct paths for images and masks within the root directory
        image_path = os.path.join(root_dir, 'images')
        mask_path = os.path.join(root_dir, 'masks')

        # To sort and retrieve lists within their respective paths
        image_files = sorted(os.listdir(image_path))
        mask_files = sorted(os.listdir(mask_path))
        assert len(image_files) == len(mask_files), f"Mismatch in number of images and masks"

        # To extend lists with full image and mask paths
        self.image_paths.extend([os.path.join(image_path, img) for img in image_files])
        self.mask_paths.extend([os.path.join(mask_path, mask) for mask in mask_files])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # To load images and masks after converting to specific formats
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask


