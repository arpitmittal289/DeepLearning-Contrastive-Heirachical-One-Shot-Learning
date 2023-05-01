# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:35:01 2023

@author: Arpit Mittal
"""

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
        for filename in glob.glob("../imagenet/train/" + file +"/*.JPEG"): #assuming gif
            img = Image.open(filename)
            img = np.array(img)
            break
        plt.title(clusterId)
        plt.imshow(img)
        plt.axis('off')
    
    
# In[ ]:

group = torch.load("10_level1 Kmeans_group.pth")
index = 1
for cluster in group.keys():
    index += 1
    view_cluster(group[cluster],index,cluster)
