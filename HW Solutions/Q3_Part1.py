# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:11:46 2020

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

def test_train_split(train_data,train_labels,Split):
    
    Train_size = round(len(train_data)*Split)
    
    # random permutation of the data in order to create a validation/test sets
    idx = np.random.permutation(train_data.shape[0])
    training_idx, test_idx = idx[:Train_size], idx[Train_size:]
    x_train, x_test = train_data[training_idx,:], train_data[test_idx,:]
    y_train, y_test = train_labels[training_idx,:], train_labels[test_idx,:]
    return(x_train,y_train,x_test, y_test)
################################## SCRIPTS ####################################

random.seed(1)

X = []
output_labels = np.zeros([441,1])

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
#=============================================================================

# Now create the test/train split of 80/20

x_train,y_train,x_test, y_test = test_train_split(X,Y,0.8)



#=============================================================================
MSE_train = []
MSE_test = []

# Now loop through various values of sigma and train the RBF network on the 
# training data. The output is calculated through predicting both sets when 
# sigma takes a broad range of values


for sigma in np.arange(0.1,5,0.1): 
    RBF_obj = RBF(x_train,x_train,y_train,sigma)
    output_test = RBF_obj.predict(x_test)
    output_train = RBF_obj.predict(x_train)
    MSE_test.append(RBF_obj.MSE(output_test,y_test))
    MSE_train.append(RBF_obj.MSE(output_train,y_train))
    print("############   Sigma = " + str(sigma) + "   ############")
    print("Training data MSE: " + str(RBF_obj.MSE(output_train,y_train)))
    print("Training data MSE: " + str(RBF_obj.MSE(output_test,y_test)))
#=============================================================================
# plot predicted vs actual MSE results as sigma varies for both test/train data

sigma = np.arange(0.1,5,0.1)
fig,ax = plt.subplots()
ax.plot(sigma,MSE_test)
ax.set(xlabel='Sigma',ylabel='MSE',
      title='Test MSE vs. Spread Parameter')
ax.grid()

fig.savefig("Q3_Part1_test.png")
plt.show()

# TRAIN PLOT
fig,ax = plt.subplots()
ax.plot(sigma,MSE_train)
ax.set(xlabel='Sigma',ylabel='MSE',
      title='Train MSE vs. Spread Parameter')
ax.grid()

fig.savefig("Q3_Part1_train.png")
plt.show()


#=============================================================================
# We see we have a massive spike in the MSE around the value of 2. Lets examine
# MSE after 2.2

fig,ax = plt.subplots()
ax.plot(sigma[26:],MSE_test[26:])
ax.set(xlabel='Sigma',ylabel='MSE',
      title='Zoomed Test MSE vs. Spread Parameter')
ax.grid()

fig.savefig("Q3_Part1_test_Zoom.png")
plt.show()

fig,ax = plt.subplots()
ax.plot(sigma[26:],MSE_train[26:])
ax.set(xlabel='Sigma',ylabel='MSE',
      title='Zoomed Train MSE vs. Spread Parameter')
ax.grid()

fig.savefig("Q3_Part1_train_Zoom.png")
plt.show()