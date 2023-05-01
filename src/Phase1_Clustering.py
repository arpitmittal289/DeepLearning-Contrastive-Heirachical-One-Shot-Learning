# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 08:46:15 2023

@author: Arpit Mittal
"""
import os
os.environ["OMP_NUM_THREADS"] = '1'
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# importing packages
from PIL import Image
import glob
import sklearn.metrics as metrics
import copy
import os
# In[ ]:

def view_cluster(files,index,clusterId):
    plt.figure(figsize = (25,25));
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        for filename in glob.glob("../imagenet/1/" + file +"/*.JPEG"): #assuming gif
            img = Image.open(filename)
            img = np.array(img)
            plt.title(clusterId)
            plt.imshow(img)
            plt.axis('off')
    
    
# # In[ ]:
# def optimalK(mean_embeddings,minRange,maxRange,incRange,level,isAgglo):
#     # K-means Clustering
#     #sse = []
#     silhouette_avg = []
#     kmeans = []
#     list_k = list(range(minRange, maxRange, incRange))
#     print("Kmeans - sse")
#     sse = []
#     for k in list_k:
#         print(k)
#         km = KMeans(n_clusters = k, init='k-means++', algorithm = 'lloyd', n_init =100)
#         km.fit(mean_embeddings)
#         kmeans.append(km)
#         sse.append(km.inertia_)
#         silhouette_avg.append(metrics.silhouette_score(mean_embeddings, km.predict(mean_embeddings), metric = "cosine"))
        
        
#     plt.plot(list_k,silhouette_avg,'bx-')
#     plt.xlabel('Values of K') 
#     plt.ylabel("Silhouette score") 
#     plt.title('Silhouette analysis For Optimal k for Kmeans ' + level)
#     plt.grid()
#     plt.show()
#     plt.savefig('Silhouette score '+level+'.png')

#     print("plotting")
#     plt.plot(list_k, sse)
#     plt.xlabel(r'Number of clusters *k*')
#     plt.ylabel('Sum of squared distance')
#     plt.title('SSE ' + level)
#     plt.grid()
#     plt.show()
#     return kmeans,silhouette_avg
    
# def kmeansclustering(mean_embeddings,embeddings_label,level,featuresize,inputsize,k,kmeansMap):
#     strin = '10_'+str(level)+'_'+str(inputsize)+'x'+str(featuresize)+'_resnet_Kmeans_k_'+str(k)
#     print("Computing - " + strin)
#     km = torch.load("10_allmean_level1 Kmeans_366x2048_resnet_k_88.pth")
    
#     groups = {}
#     got_folder_set = set()
#     got_2_names = set()
#     for file, cluster in zip(embeddings_label,km.labels_):
#         if cluster not in groups.keys():
#             groups[cluster] = set()
#             groups[cluster].add(file)
#         else:
#             groups[cluster].add(file)
#         got_folder_set.add(file)
#         if len(groups[cluster]) > 1:
#             got_2_names.add(cluster)
#     print(groups)
#     print("Saving - " + strin)
#     torch.save(km, '10_allmean_'+str(level)+'_'+str(inputsize)+'x'+str(featuresize)+'_resnet_k_'+str(k)+'.pth')
#     torch.save(groups, '10_'+level+'_group.pth')
#     print("Done - " + strin)
#     return km, groups
# # In[ ]:
    
# level_L1 = "level1"
# mean_embeddings_L1 = torch.load("00_mean_embeddings_resnet.pth")
# embeddings_label_L1 = torch.load("00_embeddings_folder_label_resnet.pth")
# inputsize_L1 = mean_embeddings_L1.shape[0]
# featuresize_L1 = mean_embeddings_L1.shape[1]

# # To find optimal K
# isAgglo = False
# kmeansMap1,silhouette_avg = optimalK(mean_embeddings_L1,80,120,1,level_L1,isAgglo)

# Kmeans_kvalue_L1 = 88
# km_L1,group_L1 = kmeansclustering(mean_embeddings_L1,embeddings_label_L1,level_L1+" Kmeans",featuresize_L1,inputsize_L1,Kmeans_kvalue_L1,kmeansMap1)
# #aglo_L1,aglo_group_L1 = kmeansclustering(mean_embeddings_L1,embeddings_label_L1,level_L1+" Agglo",featuresize_L1,inputsize_L1,agglo_kvalue_L1,AggloMap1)
# metrics.silhouette_score(mean_embeddings_L1, km_L1.predict(mean_embeddings_L1), metric = "cosine")
# # In[ ]:
  
# level_L2 = "level2"
# mean_embeddings_L2 = copy.deepcopy(km_L1.cluster_centers_)
# embeddings_label_L2 = copy.deepcopy(km_L1.labels_)
# inputsize_L2 = mean_embeddings_L2.shape[0]
# featuresize_L2 = mean_embeddings_L2.shape[1]

# isAgglo = False
# kmeansMap2 = optimalK(mean_embeddings_L2,2,20,1,level_L2,isAgglo)
# Kmeans_kvalue_L2 = 8
# km_L2,group_L2 = kmeansclustering(mean_embeddings_L2,embeddings_label_L2,level_L2+" Kmeans",featuresize_L2,inputsize_L2,Kmeans_kvalue_L2,kmeansMap2)

# In[ ]:
# To find optimal K
group = torch.load("10_level1 Kmeans_group.pth")
index = 1
for cluster in group.keys():
    index += 1
    view_cluster(group[cluster],index,cluster)

