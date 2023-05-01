import torch
import torchvision.models as models
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import json
import glob
import sklearn.metrics as metrics
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
from statistics import mean as average
warnings.filterwarnings("ignore")
# In[]:
print("PRETRAINED MODEL USING ONE SHOT LEARNING")
data_embeddings = torch.load('./mapping.pth', map_location=torch.device('cpu'))
data_embeddings = data_embeddings.reset_index()
    

root_src = "../imagenet/"
train_src = '1'
test_src = '1_2'
batchSize = 64
class_to_cluster_embedding={}

print("Creating Target Embedding Data")
for index, row in data_embeddings.iterrows():
    class_folder = row['Label'] 
    class_to_cluster_embedding[class_folder] = {}
    class_to_cluster_embedding[class_folder][0] = row['Level0_Embeddings']
    class_to_cluster_embedding[class_folder][1] = row['Level1_Embeddings']
    class_to_cluster_embedding[class_folder][2] = row['Level2_Embeddings']
    
# In[]:     
class ImageNetDataset(Dataset):
    def __init__(self, root, split, level, transform=None):
        self.samples = []
        self.target_labels = []
        self.target_embeddings = []
        self.transform = transform
        self.syn_to_class = {}
        self.level = level
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        samples_dir = os.path.join(root, split)
        for entry in os.listdir(samples_dir):
#             print('Entry:',entry)
            syn_id = entry
            if syn_id not in class_to_cluster_embedding:
                continue
            
            sample_path = os.path.join(samples_dir, entry)
            for filename in glob.glob(sample_path+"/*"): #assuming gif
                im=Image.open(filename).convert('RGB')
                sample = self.transform(im)
            
                target_label = self.syn_to_class[syn_id]
                
                target_embedding = class_to_cluster_embedding[syn_id][self.level]
                    
                self.samples.append(sample)
                self.target_labels.append(torch.tensor(target_label))
                self.target_embeddings.append(torch.tensor(target_embedding))
                
        zipped = list(zip(self.samples, self.target_embeddings))
        random.shuffle(zipped)
        self.samples, self.target_embeddings = zip(*zipped)

#               print('Target 2:',target)
#         print(self.targets)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            return self.samples[idx], self.target_embeddings[idx]

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, image_embeddings, label_embeddings):

        distances = torch.cdist(image_embeddings, label_embeddings)
        negative_distances = (self.margin - distances).clamp(min=0)
        positive_distances = distances
        loss = torch.cat([positive_distances, negative_distances], dim=1).mean()
        return loss

# In[]: 
    
## LEVEL 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available
print(f"Using device: {device}")

    
data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

# In[]: 
batchStep = 10
    
def trainEncoderForLevel(model, loss_fn, optimizer, scheduler, level, num_epochs, train_data, val_data, test_data):
    print("Training ")
    train_loss_lst = []
    train_loss_per_Batch = []
    val_cosin_sim = []
    val_accuracy = []
    model.to(device)
    num_epochs = 21
    for epoch in range(num_epochs):
        for i, (images, embeddings) in enumerate(train_data):
            target_embeddings = torch.unsqueeze(torch.unsqueeze(embeddings, dim=2), dim=2)
            image_embeddings = model(images.to(device))
            train_loss = loss_fn(image_embeddings.to(device), target_embeddings.to(device))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_lst.append(train_loss.item())
        if epoch > 0 and epoch%10 == 0:
            print('Epoch [{}/{}], Train Loss: {:.4f}'
              .format(epoch, num_epochs, np.array(train_loss_lst).mean()))
        
        with torch.no_grad():
            cosine_sim = []
            cont_loss = []
            for images, embeddings in test_data:
                predicted_embeddings = model(images.to(device))
                target_embeddings = torch.unsqueeze(torch.unsqueeze(embeddings, dim=2), dim=2)
                cos_sim = metrics.pairwise.cosine_similarity(predicted_embeddings.flatten(start_dim=1).cpu(), embeddings.flatten(start_dim=1).cpu())
                cosine_sim.append(cos_sim[0][0])
                loss = loss_fn(predicted_embeddings.cpu(), target_embeddings.cpu())
                cont_loss.append(loss)
            val_cosin_sim.append(average(cosine_sim))
            val_accuracy.append(1-np.array(cont_loss).mean())

    plotTrainValCurve(val_accuracy, val_cosin_sim, level)
        
    print("Testing ")
    cosine_sim = []
    cont_loss = []
    with torch.no_grad():
        for images, embeddings in test_data:
            predicted_embeddings = model(images.to(device))
            target_embeddings = torch.unsqueeze(torch.unsqueeze(embeddings, dim=2), dim=2)
            cos_sim = metrics.pairwise.cosine_similarity(predicted_embeddings.flatten(start_dim=1).cpu(), embeddings.flatten(start_dim=1).cpu())
            cosine_sim.append(cos_sim[0][0])
            loss = loss_fn(predicted_embeddings.cpu(), target_embeddings.cpu())
            cont_loss.append(loss)
    
    torch.save(model,"L"+str(level)+"_ONESHOT_PRETRAINED_MODEL")

    print("Average Level " + str(level) + " Test cosine Similarity:", np.array(cosine_sim).mean())
    print("Average Level " + str(level) + " Test loss:", np.array(cont_loss).mean())  
        
    return model, train_loss_per_Batch, val_cosin_sim, cosine_sim, cont_loss
    
def plotTrainValCurve(val_accuracy, val_cosin_sim, level):
    num_batch = range(len(val_cosin_sim))
    plt.figure(figsize = (6,6))
    plt.plot(num_batch,val_cosin_sim, label="Cosine Similarity Loss")
    plt.plot(num_batch,val_accuracy, label="Accuracy")
    plt.xlabel('Epochs') 
    plt.ylabel("Loss") 
    plt.title('Val Cosine Similarity Curve for Training One Shot Level ' + str(level))
    plt.legend()
    plt.show()
    plt.savefig("OneShot_Cosine_Sim_Curve "+str(level))
    
# In[]: 
'''
'''

print("-------------------------")
print("Starting LEVEL 0")
print("Creating Traininig Data")
train_dataset_l0 = ImageNetDataset(root_src, train_src, 0, data_transforms)
train_dataloader_l0 = DataLoader(
            train_dataset_l0,
            batch_size=batchSize, # may need to reduce this depending on your GPU 
            shuffle=False,
        )
'''
print("Creating Validation Data")
val_dataset_l0 = ImageNetDataset(root_src, val_src, 0, data_transforms)
val_dataloader_l0 = DataLoader(
            val_dataset_l0,
            batch_size=batchSize, # may need to reduce this depending on your GPU 
            shuffle=False,
        )
'''

print("Creating Testing Data")
test_dataset_l0 = ImageNetDataset(root_src, test_src, 0, data_transforms)
test_dataloader_l0 = DataLoader(
            test_dataset_l0,
            batch_size=1, # may need to reduce this depending on your GPU 
            shuffle=False,
        )
model = models.resnet152(pretrained=True)

model_resnet152_l0 = torch.nn.Sequential(*list(model.children())[:-1])

for param in model_resnet152_l0.parameters():
    param.requires_grad = True
model_resnet152_l0[-1].requires_grad = True
    
contrastive_loss_fn_l0 = ContrastiveLoss()
optimizer_l0 = torch.optim.Adam(model_resnet152_l0.parameters(), lr= 0.01)
scheduler_l0 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l0, mode='min', patience=3, factor=0.9)
val_dataloader_l0 = None
num_epochs = 2
model_resnet152_l0, train_loss_l0, val_loss_l0, cosine_sim_l0, cont_loss_l0 = trainEncoderForLevel(model_resnet152_l0, contrastive_loss_fn_l0, optimizer_l0, scheduler_l0, 0, num_epochs, train_dataloader_l0, val_dataloader_l0, test_dataloader_l0)
# In[]: 
## LEVEL 1
print("-------------------------")
print("Starting LEVEL 1")
print("Creating Traininig Data")
train_dataset_l1 = ImageNetDataset(root_src, train_src, 1, data_transforms)
train_dataloader_l1 = DataLoader(
            train_dataset_l1,
            batch_size=batchSize, # may need to reduce this depending on your GPU 
            shuffle=False,
        )
print("Creating Testing Data")
test_dataset_l1 = ImageNetDataset(root_src, test_src, 1, data_transforms)
test_dataloader_l1 = DataLoader(
            test_dataset_l1,
            batch_size=1, # may need to reduce this depending on your GPU 
            shuffle=False,
        )
'''
print("Creating Validation Data")
val_dataset_l1 = ImageNetDataset(root_src, val_src, 1, data_transforms)
val_dataloader_l1 = DataLoader(
            val_dataset_l1,
            batch_size=batchSize, # may need to reduce this depending on your GPU 
            shuffle=False,
        )
'''
model = models.resnet152(pretrained=True)
model_resnet152_l1 = torch.nn.Sequential(*list(model.children())[:-1])
for param in model_resnet152_l1.parameters():
    param.requires_grad = True

contrastive_loss_fn_l1 = ContrastiveLoss()
optimizer_l1 = torch.optim.Adam(model_resnet152_l1.parameters(), lr= 0.01)
scheduler_l1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l1, mode='min', patience=3, factor=0.9)
num_epochs = 2
val_dataloader_l1 = None
model_resnet152_l1, train_loss_l1, val_loss_l1, cosine_sim_l1, cont_loss_l1 = trainEncoderForLevel(model_resnet152_l1, contrastive_loss_fn_l1, optimizer_l1, scheduler_l1, 1, num_epochs, train_dataloader_l1 , val_dataloader_l1, test_dataloader_l1)

# In[ ]: 

## LEVEL 2
print("-------------------------")

print("Starting LEVEL 2")
print("Creating Traininig Data")
train_dataset_l2 = ImageNetDataset(root_src, train_src, 2, data_transforms)
train_dataloader_l2 = DataLoader(
            train_dataset_l2,
            batch_size=batchSize, # may need to reduce this depending on your GPU 
            shuffle=False,
        )
print("Creating Testing Data")
test_dataset_l2 = ImageNetDataset(root_src, test_src, 2, data_transforms)
test_dataloader_l2 = DataLoader(
            test_dataset_l2,
            batch_size=1, # may need to reduce this depending on your GPU 
            shuffle=False,
        )

'''
print("Creating Validation Data")
val_dataset_l2 = ImageNetDataset(root_src, val_src, 2, data_transforms)
val_dataloader_l2 = DataLoader(
            val_dataset_l2,
            batch_size=batchSize, # may need to reduce this depending on your GPU 
            shuffle=False,
        )
'''
model = models.resnet152(pretrained=True)
model_resnet152_l2 = torch.nn.Sequential(*list(model.children())[:-1])
for param in model_resnet152_l2.parameters():
    param.requires_grad = True

contrastive_loss_fn_l2 = ContrastiveLoss()
optimizer_l2 = torch.optim.Adam(model_resnet152_l2.parameters(), lr= 0.01)
scheduler_l2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_l2, mode='min', patience=3, factor=0.9)
num_epochs = 2
val_dataloader_l2 = None
model_resnet152_l2, train_loss_l2, val_loss_l2, cosine_sim_l2, cont_loss_l2  = trainEncoderForLevel(model_resnet152_l2, contrastive_loss_fn_l2, optimizer_l2, scheduler_l2, 2, num_epochs, train_dataloader_l2, val_dataloader_l2, test_dataloader_l2)
# In[ ]: 
'''
'''


'''
total_loss += loss.item()

dist = metrics.pairwise.cosine_distances(predicted_embeddings.cpu(), embeddings.cpu())
# Compute the number of correct predictions
diff = (predicted_embeddings.to(device) - embeddings.to(device))

dists = torch.norm(diff, dim=1).mean(axis=1)
preds = (dists < 0.5).long()
total_correct += preds.sum().item()

total_samples += labels.size(0)

# Compute the average loss and accuracy
avg_loss = total_loss / total_samples
accuracy = total_correct / total_samples

print("Average loss:", avg_loss)
print("Accuracy:", accuracy)
'''