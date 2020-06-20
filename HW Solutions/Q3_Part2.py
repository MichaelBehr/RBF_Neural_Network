# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:31:59 2020

@author: Michael
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

########################### CLASS FOR RBF MODEL ###############################

# we are going to create a class to create our RBF model
class RBF:
    def __init__(self, input_data, RBF_clusters, output_labels,sigma):
        
        # store the input data, and output_labels
        self.input_data = input_data
        self.output_labels = output_labels
        self.clusters = RBF_clusters      
        self.sigma = sigma
        
        # Generate G with the inputs and our clusters
        self.G = self.generate_hidden()
        
        # Generate W with the equation (GTG)-1GT * D
        G1 = np.dot(np.transpose(self.G),self.G)
        G2 = np.linalg.inv(G1)
        G3 = np.dot(G2,np.transpose(self.G))
        self.W = np.dot(G3,self.output_labels)

    def activation(self,x,g):
        return(np.exp(-self.euclidean_distance(x,g)**2/2*self.sigma**2))
    
    def euclidean_distance(self,p,q):
        return(np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2))
        
    def generate_hidden(self):
        G = np.zeros([len(self.input_data),len(self.clusters)])
        for i in range(len(self.input_data)):
            for j in range(len(self.clusters)):
                G[i][j] = self.activation(self.input_data[i],self.clusters[j])
        return(G)

    def predict(self,x_test):
        self.input_data = x_test
        self.G = self.generate_hidden()
        predictions = np.dot(self.G,self.W)
        
        return(predictions)
    
    def get_acc(self, y_pred,y):
        Accuracy = 0
        for x,y in zip(y_pred, y):
            if np.argmax(x) == np.argmax(y):
                Accuracy = Accuracy + 1
        return(Accuracy/len(y_pred)*100)
        
        
    def MSE(self,y_pred,y):
        return(((y_pred - y)**2).mean(axis=0)[0])
    
    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
################################## SCRIPTS ####################################

random.seed(1)

X = []
X1 = []
output_labels1 = np.zeros([441,1])
output_labels = np.zeros([441,1])

# =============================================================================
 ## uniform random implementation
for x in range(441):
    i = random.uniform(-2,2)
    j = random.uniform(-2,2)
    X1.append([i,j])
    if((i**2+j**2)<=1):
        output_labels1[x][0] = 1
    if((i**2+j**2)>1):
        output_labels1[x][0] = -1
# =============================================================================
    

#=============================================================================
 ## implementation using equation from assignment
X = []
x = 0
for i in range(21):
    for j in range(21):
        x1 = -2+0.2*i
        x2 = -2+0.2*j
        X.append([x1,x2])
        if((x1**2+x2**2)<=1):
            output_labels[x][0] = 1
        if((x1**2+x2**2)>1):
            output_labels[x][0] = -1
        x = x + 1

idx = np.random.permutation(len(X))
X = np.array(X)[idx]
Y = output_labels[idx]
## check dupes
 
# for i in np.arange(0,440):
#     for j in np.arange(i+1,441):
#         if X2[i]==X2[j]:
#             print("Duplicate found at position i= : ", i)
#             print("\n j=",j)
             
#=============================================================================

# Keep sigma at 5 from part 1
MSE = []
sigma = 1.5


###############################################################################
# Calculate clusters through random selection
Num_Clusters = 150
idx = np.random.permutation(len(X))[:Num_Clusters]
clusters1 = [X[i,:] for i in idx]

# Calculate clusters through k-means
clusters2 = KMeans(n_clusters=Num_Clusters,random_state=1).fit(X).cluster_centers_


MSE_rand = []
MSE_Kmeans = []

for sigma in np.arange(0.5,3,0.1): 
    RBF_Rand = RBF(X,clusters1,Y,sigma)
    RBF_Kmeans = RBF(X,clusters2,Y,sigma)

    output_Rand = RBF_Rand.predict(X)
    output_Kmeans = RBF_Kmeans.predict(X)


    MSE_rand.append(RBF_Rand.MSE(output_Rand,Y))
    MSE_Kmeans.append(RBF_Kmeans.MSE(output_Kmeans,Y))
    
sigma = np.arange(0.5,3,0.1)
fig,ax = plt.subplots()
ax.plot(sigma,MSE_rand)
ax.plot(sigma,MSE_Kmeans)
ax.set(xlabel='Sigma',ylabel='MSE',
      title='MSE vs. Spread Parameter')
ax.grid()

fig.savefig("Q3_Part2_Zoom.png")
plt.show()