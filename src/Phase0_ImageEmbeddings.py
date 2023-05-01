#!/usr/bin/env python
# coding: utf-8
# In[ ]:


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from PIL import Image
import os
from torch.utils.data import Dataset
import json
import glob
import torchvision
import torchvision.transforms as transforms
import math

root_src = "../imagenet/"
train_data_src = 'train'

train_set = set()
def is_int(v):
    return type(v) is int

isgrayscale = []

syn_to_class = {}
with open(os.path.join(root_src, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                syn_to_class[v[0]] = int(class_id)

samples_dir = os.path.join(root_src, train_data_src)
num_class = os.listdir(samples_dir)


class ImageNetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.label = ""
        self.folder_name = ""
        samples_dir = os.path.join(root, split)
        for entry in os.listdir(samples_dir):
                syn_id = entry
                if syn_id in train_set:
                    continue
                
                print(syn_id)
                target = syn_to_class[syn_id]
                self.folder_name = syn_id
                self.label = target
                train_set.add(entry)
                sample_path = os.path.join(samples_dir, entry)
                for filename in glob.glob(sample_path+"/*.jpg"): #assuming gif
                    im=Image.open(filename)
                    if type(im.getpixel((0,0))) is int:
                        isgrayscale.append((filename,target))
                        continue
                    self.samples.append(self.transform(im))
                    self.targets.append(target)
                    
                for filename in glob.glob(sample_path+"/*.JPEG"): #assuming gif
                    im=Image.open(filename)
                    if type(im.getpixel((0,0))) is int:
                        isgrayscale.append((filename,target))
                        continue
                    self.samples.append(self.transform(im))
                    self.targets.append(target)
                break
                    
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            return self.samples[idx], self.targets[idx]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available
print(f"Using device: {device}")

        
# In[]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# Define the validation data loader
data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
# In[ ]:
print('----------')
import torch
import torch.nn as nn
import torchvision.models as models

resnet152 = models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
modules=list(resnet152.children())[:-1]
resnet152=nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False

resnet152.to(device)
# Remove final classification layer
#efficientnet_model._fc = torch.nn.Identity()
# In[ ]:
total_embeddings = []
total_embeddings_label = []
import numpy as np
while True:
    try:
        embeddings = []
        labels = []
        if len(train_set) == len(num_class):
            break
        
        train_dataset = ImageNetDataset(root_src, train_data_src, data_transforms)
        train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=64, # may need to reduce this depending on your GPU 
                    shuffle=False
                )
        
        with torch.no_grad():
            for x, y in tqdm(train_dataloader):
                activations = resnet152(x.cuda())
                embeddings.extend(torch.nn.functional.normalize(activations.cpu().flatten(start_dim=1)))
                labels.extend(np.array(y))
                
        embeddings_array = np.array([embed.numpy() for embed in embeddings])
        torch.save(embeddings_array, '../imagenet_embeddings_resnet/'+str(train_dataset.folder_name)+'_'+str(train_dataset.label)+'.pth')
        torch.save(train_set, 'train_set_resnet.pth')
        total_embeddings.extend(embeddings_array)
        total_embeddings_label.extend(labels)
    except Exception as e:
        print("An exception occurred")
        print(e)
        
    
    #torch.save(np.array(total_embeddings), '10_imagenet_resnet_embeddings_final.pth')
    #torch.save(np.array(total_embeddings_label), '10_imagenet_resnet_embeddings_lables_final.pth')
    '''
    totalEmbeddings.extend(mean_embedding)
    labels.extend(y.cpu()[:int(embeddings_array.shape[0]/mean_factor)*mean_factor].tolist())
    '''
 
mean_embeddingsPath = "../imagenet_embeddings_resnet/"
mean_embeddings = []
embeddings_label = []
embeddings_folder_label = []
for class_id in os.listdir(mean_embeddingsPath):
        print(class_id)
        embedding_path = mean_embeddingsPath + class_id 
        embeddings = torch.load(embedding_path)
        
        order = np.argsort(np.mean(embeddings,axis=1))
        embeddings_sorted = embeddings[order]
        mean_factor = len(embeddings_sorted)
        truncated_length = math.floor(len(embeddings_sorted)/mean_factor) * mean_factor

        embeddings_sorted_truncated = embeddings_sorted[:truncated_length]
        
        embeddings_mean_at_factor = embeddings_sorted_truncated.reshape(int(truncated_length / mean_factor) , mean_factor , 2048)
        
        embeddings_mean = embeddings_mean_at_factor.mean(axis = 1)
        
        mean_embeddings.extend(embeddings_mean)
        for embed in range(len(embeddings_mean)):
            embeddings_label.append(class_id.split('.')[0].split('_')[1])
            embeddings_folder_label.append(class_id.split('_')[0])
        
mean_embeddings = np.array(mean_embeddings)
embeddings_label = np.array(embeddings_label)
embeddings_folder_label = np.array(embeddings_folder_label)

torch.save(mean_embeddings, '00_mean_embeddings_resnet.pth')
torch.save(embeddings_label, '00_embeddings_label_resnet.pth')
torch.save(embeddings_folder_label, '00_embeddings_folder_label_resnet.pth')
print("Embeddings Done")
print(len(train_set))
