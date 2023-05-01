# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:44:13 2023

@author: Arpit Mittal
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
# importing packages
from PIL import Image
import glob
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")
import copy 

embedding = torch.load("new_unseen_embeddings_test.pth")
km = torch.load("10_allmean_level1 Kmeans_366x2048_resnet_k_88.pth")


cluster_centre = {}
exp = 1
mul = 1
embedding_list = []
embedding_map = {}
for i, unseen_class in enumerate(embedding.keys()):
    embedding_list.append(np.array(embedding[unseen_class][0].flatten().cpu()))
    embedding_map[unseen_class] = i
    
embedding_arr = np.array(torch.tensor(embedding_list))
km_cluster_centers = km.cluster_centers_

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler()
embedding_arr = scaler.fit_transform(embedding_arr)
km_cluster_centers = scaler.transform(km_cluster_centers)

emedding_old = copy.deepcopy(embedding)
for i, unseen_class in enumerate(emedding_old.keys()):
    embedding[unseen_class] = embedding_arr[embedding_map[unseen_class]]
    
for i, clust in enumerate(km_cluster_centers):
    cluster_centre[i] = torch.unsqueeze(
        torch.unsqueeze(torch.tensor(clust), dim=1), dim=2)

# In[]:
    
assigned_clust = {}
for unseen_class in embedding.keys():
    unseen_class_embedding = embedding[unseen_class]
    assigned_clust[unseen_class] = 0
    maxSim = -1
    point2 = torch.pow(torch.unsqueeze(torch.tensor(unseen_class_embedding.flatten()), dim=0).cpu() * mul, exp)
    for clust_key in cluster_centre.keys():
        point1 = torch.pow(torch.unsqueeze(torch.tensor(cluster_centre[clust_key]).flatten(), dim=0).cpu() * mul, exp)        
        sim = metrics.pairwise.cosine_distances(point1, point2)
        if sim[0][0] > maxSim:
            assigned_clust[unseen_class] = str(clust_key)
            maxSim = sim[0][0]

print(assigned_clust)
# In[]:
# assigned_clust = {}
# for unseen_class in embedding.keys():
#     unseen_class_embedding = embedding[unseen_class]
#     assigned_clust[unseen_class] = 0
#     maxSim = -1
#     point2 = torch.pow(torch.unsqueeze(torch.tensor(unseen_class_embedding.flatten()), dim=0).cpu() * mul, exp)
#     skipClust = []
#     for clust_key in cluster_centre.keys():
#         point1 = torch.pow(torch.unsqueeze(torch.tensor(cluster_centre[clust_key]).flatten(), dim=0).cpu() * 10, exp)        
#         sim = metrics.pairwise.paired_cosine_distances(point1, point2)
#         if sim[0] > maxSim:
#             assigned_clust[unseen_class] = str(clust_key)
#             maxSim = sim[0]

# print(assigned_clust)
# # In[]:
    
# embedding_sim  = {}
# for unseen_class in embedding.keys():
#     unseen_class_embedding = embedding[unseen_class][0]
#     assigned_clust[unseen_class] = 0
#     mindist = 100
#     maxSim = -1
#     for clust_key in embedding.keys():
#         clus_embedding = embedding[clust_key][0]
#         point1 = torch.pow(torch.unsqueeze(torch.tensor(clus_embedding.flatten()), dim=0).cpu() * mul , exp)
#         point2 = torch.pow(torch.unsqueeze(torch.tensor(unseen_class_embedding.flatten()), dim=0).cpu() * mul, exp)
#         sim = metrics.pairwise.cosine_similarity(point1, point2)
#         if sim[0][0] > maxSim:
#             maxSim = sim[0][0]
#             embedding_sim[unseen_class] = clust_key +"_"+ str(maxSim)
            
# print(embedding_sim)