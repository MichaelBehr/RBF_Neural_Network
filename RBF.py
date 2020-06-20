# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:15:09 2020

@author: Michael
"""

import numpy as np
import random
import matplotlib.pyplot as plt

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