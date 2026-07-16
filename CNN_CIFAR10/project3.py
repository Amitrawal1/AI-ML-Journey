#!/usr/bin/env python
# coding: utf-8

# In[38]:


import torch 
import torch.nn as nn
import torch.optim as optim

import torchvision 
from torchvision.datasets import CIFAR10


# In[39]:


#Datasets and DataLoader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# image --TO-- scale (0,1) --TO-- nomrmalize (-1,1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = CIFAR10(root = "./Data", train=True, download=False, transform=transform)
testset = CIFAR10(root = "./Data", train=False, download=False, transform=transform)


# In[40]:


trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64)


# In[41]:


# Build CNN 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.Conv_layer = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # kernel size = 2, stride = 2


            nn.Conv2d(32,64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2) 
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(4*4*128, 256),
            nn.ReLU(),
            nn.Linear(256,10)
        )

    def forward(self,x):
        x = self.Conv_layer(x)
        x = x.view(x.size(0), -1) # flattening
        x = self.fc_layer(x)

        return x


# In[42]:


model = CNN()


# In[43]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# In[46]:


### Train CNN 
epochs = 10

for epoch in range(epochs):
    epoch_training_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()
        
        output = model.forward(images) #forwar prop
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        epoch_training_loss += loss.item()

    print(f"epoch={epoch+1}/{epochs} & loss={epoch_training_loss/len(trainloader)}")


# In[51]:


correct_labels = 0
total_labels = 0

model.eval()

with torch.no_grad():
    for images, labels in testloader:
        outputs = model.forward(images)
        _, predict = torch.max(outputs, 1)

        correct_labels += (predict == labels).sum().item()
        total_labels += labels.size(0)

print(f"Accuracy = {correct_labels/total_labels * 100}")
        

