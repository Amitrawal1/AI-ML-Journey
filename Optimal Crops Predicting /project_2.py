#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd 
import numpy as np


# In[79]:


df = pd.read_csv("Crop_Data.csv")
df.head()


# In[80]:


df.shape


# In[82]:


X = df.drop("Crop", axis=1)
y = df["Crop"]
X.shape


# In[83]:


from sklearn.preprocessing import StandardScaler, LabelEncoder

le = LabelEncoder()
y =le.fit_transform(y)


# In[84]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

y_train


# In[85]:


scaler = StandardScaler()

X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# In[86]:


# Now  work on the  ANN part 
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# In[87]:


X_train_tensor = torch.tensor(X_trained_scaled, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.long)

X_test_tensor = torch.tensor(X_test_scaled, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test, dtype = torch.long)


# In[88]:


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[107]:


class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 22)
        )
        
    def forward(self, X):
        return self.model(X)
        
        


# In[108]:


# Loss and optim
model = ANN()

criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# In[109]:


epochs = 100

for epoch in range(epochs):
    model.train()

    running_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        
        output =  model(xb)
        loss = criteria(output, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss/len(train_loader)

    print(f" epoch = {epoch+1}/{epochs}, loss = {train_loss}")


# In[ ]:


# For 1 Hidden Layer,  32 nodes loss = 0.030101546602831646
# For 1 Hidden Layer,  64 nodes loss = 0.02101434163156558
# For 2 Hidden Layers, 32 nodes, loss = 0.019440602365648374
# for 2 Hidden Layers, 64 nodes, loss = 0.007916331916666505


# In[110]:


# Evaluate Phase
model.eval()

total = 0
correct = 0

with torch.no_grad():
    for xb, yb in test_loader:
        output = model(xb)
        
        # Yahan humne class ka index nikala aur 'predicted' mein save kiya
        _, predicted = torch.max(output, 1)

        # Fix 1: 'predict' ki jagah 'predicted' likha
        # Fix 2: yb ko yb.view(-1) kiya taaki size hamesha match ho
        correct += (predicted == yb.view(-1)).sum().item()
        
        total += yb.size(0)

# Final Accuracy Print
print(f"Accuracy: {(correct/total)*100:.2f}%")


# In[ ]:


# For 1 Hidden Layer,  32 nodes Accuracy: 97.05%
# For 1 Hidden Layer,  64 nodes Accuracy: 96.82%
# For 2 Hidden Layers, 32 nodes Accuracy: 97.27%
# For 2 Hidden Layers, 64 nodes Accuracy: 97.73%

