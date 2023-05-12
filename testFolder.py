import sys
import torchvision
import torch.nn as nn
import torch
import os
import PIL
from torcheval.metrics.functional import r2_score, mean_squared_error
from torch.utils.data import Dataset
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_excel(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.count = 0
        self.imgsArr = []
        self.labelArr = []
        for idx in self.img_labels.index:
            imgPath = os.path.join(self.img_dir,self.img_labels.iloc[idx,1])
            if os.path.exists(imgPath):
                self.count+=1
                self.imgsArr.append(imgPath)
                self.labelArr.append(self.img_labels.iloc[idx,2])


    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        img_path = self.imgsArr[idx]
        image = PIL.Image.open(img_path)
        T1 = torchvision.transforms.Resize((250,250))
        T2 = torchvision.transforms.ToTensor()
        image = T1(image)
        image = T2(image)
        image = image[:-1]
        assert(image.shape[0]==3)
        label = self.labelArr[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__=="__main__":
    modelPath = sys.argv[2]
    inputPath = sys.argv[4]
    outputPath = sys.argv[6]

    annotationsPath = os.path.join(inputPath, "annotations.xlsx")
    imgsPath = os.path.join(inputPath, "images")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    testDataset = CustomImageDataset(annotationsPath, imgsPath)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=32, shuffle=False)

    with open(outputPath, 'w') as outFile:
        outFile.write("Model,R2Score,MSE\n")

        directory = os.fsencode(modelPath)
        for filename in os.listdir(directory):
            modelPath = os.fsdecode(os.path.join(directory, filename))

            model_vgg = torch.load(modelPath,map_location=device)
            model_vgg.eval()

            outputs = torch.zeros(len(testDataset)).to(device)
            targets = torch.zeros(len(testDataset)).to(device)
            idx = 0
            for image,label in testLoader:
                image = image.to(device)
                label = label.to(device)
                output = model_vgg(image).flatten()
                for i in range(image.shape[0]):
                    outputs[idx] = output[i]
                    targets[idx] = float(label[i])
                    idx+=1

            r2Score = r2_score(outputs, targets)
            mse = mean_squared_error(outputs, targets)

            outFile.write(f"{os.path.basename(modelPath)},{r2Score:.4f},{mse:.4f}\n")

