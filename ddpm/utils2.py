import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader



def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

class ImageAndMaskDataset(Dataset):
    def __init__(self, img_dir, msk_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_folder = torchvision.datasets.ImageFolder(img_dir, transform=None)
        self.mask_folder = msk_dir  

    def __len__(self):
        return len(self.image_folder)



    def __getitem__(self, idx):

        img_path, label = self.image_folder.imgs[idx]
        mask_name = os.path.basename(img_path).replace(".jpg", "_segmentation.png") 

        mask_path = os.path.join(self.mask_folder, str(label), mask_name)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask, label

class MaskDataset(Dataset):

    def __init__(self, msk_dir, transform=None):
        self.msk_dir = msk_dir
        self.transform = transform
        self.mask_folder = torchvision.datasets.ImageFolder(msk_dir, transform=None)

    def __len__(self):
        return len(self.mask_folder)    



    def __getitem__(self, idx):

        mask_path, label = self.mask_folder.imgs[idx]
        mask = Image.open(mask_path)    
        name = os.path.basename(mask_path)
        if self.transform:
            mask = self.transform(mask)
        return mask, label, name



def get_mask_data(args):

    transform = torchvision.transforms.Compose([

        torchvision.transforms.Resize((args.image_size, args.image_size)), 

        torchvision.transforms.ToTensor() 

    ])

    dataset = MaskDataset(args.mask_path, transform=transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) 
    return dataloader        

def get_data(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size, args.image_size)), 
        torchvision.transforms.ToTensor() 
    ])
    dataset = ImageAndMaskDataset(args.image_path, args.mask_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  
    return dataloader


def setup_logging(run_name):

    os.makedirs("models", exist_ok=True)

    os.makedirs("results", exist_ok=True)

    os.makedirs(os.path.join("models", run_name), exist_ok=True)

    os.makedirs(os.path.join("results", run_name), exist_ok=True)



def model_init(model_weight, model_bias=None):
    if not model_weight:
        torch.nn.init.xavier_normal(model_weight)
    if not model_bias:
        torch.nn.init.zeros_(model_bias)

        
