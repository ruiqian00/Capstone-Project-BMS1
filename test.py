from data import main_transformations
from torch.utils.data import DataLoader,Dataset
from models import create_model
import torch
from torchvision.io import read_image
import os
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
import torchvision.transforms.functional as TF
import argparse

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')  
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_dir', type=str, default='./test', help='Directory to test images')
    args = parser.parse_args()

    transform = main_transformations()
    test_dir = './test'
    dataset = ImageDataset(args.test_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'swin_transformer'
    model = create_model(model_name, num_classes=4)
    model.load_state_dict(torch.load(f'{model_name}_best_weights.pth'))
    model = model.to(device)

    model.eval()  

    results = []

    
    for images, filenames in dataloader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_classes = predicted.cpu().numpy()

        for filename, pred in zip(filenames, predicted_classes):
            results.append([filename, pred])

    df = pd.DataFrame(results, columns=['Filename', 'Class'])
    class_dict = {0: 'clear', 1: 'empty', 2: 'pure', 3: 'rest'}
    df['Class'] = df['Class'].map(class_dict)
    df.to_csv('prediction_results.csv', index=False)
