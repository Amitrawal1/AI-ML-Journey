#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np


# In[35]:


# Data Loading 
df = pd.read_csv("./Data/data_1a.csv")
df.head


# In[36]:


df.isnull().sum()


# In[37]:


X = df.drop("PE", axis = 1)
y = df["PE"]


# In[38]:


X.head()
y.head()


# In[39]:


# Split data into Training and Testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

df.shape


# In[40]:


# Apply Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[10]:


X_train_scaled

X_test_scaled


# In[41]:


# Convert into Tensor
import torch 
import torch.nn as nn
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)


# In[42]:


# dataset and Dataloader
from torch.utils.data import TensorDataset, DataLoader

Train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
Test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(Train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(Test_dataset, batch_size=32)


# In[43]:


# Make a ANN model
class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()

        self.model = nn.Sequential(
            # 1st hidden Layer
            nn.Linear(X_train.shape[1], 6),
            nn.ReLU(),

            #2nd hidden Layer
            nn.Linear(6, 6),
            nn.ReLU(),

            #3rd hidden Layer
            nn.Linear(6, 6),
            nn.ReLU(),

            #Output Layer
            nn.Linear(6, 1),
        )
    def forward(self, x):
        return self.model(x)


# In[44]:


# Loss, Optimizer 
import torch.optim as optim
model = ANN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())


# In[46]:


# Train the ANN
train_losses = []
valid_losses = []

best_val_loss = float("inf")

epochs = 100

for epoch in range(epochs): 
    model.train()
    running_loss = 0.0   # total training loss in 1 epoch

    for xb,yb in train_loader:
        optimizer.zero_grad()  # fresh start gradient for each batch

        output = model(xb) 
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()    # item() = convert tensor to  py float 

    epoch_training_loss = running_loss/len(train_loader)
    train_losses.append(epoch_training_loss)


    # Validation
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for xb,yb in test_loader:
            
            output = model(xb)
            loss = criterion(output, yb)
            running_val_loss += loss

    epoch_val_loss = running_val_loss / len(test_loader)
    valid_losses.append(epoch_val_loss)


    print(f"epoch {epoch+1}/ {epochs} ==> Train_loss = {epoch_training_loss} && Val_loss = {epoch_val_loss}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_model.pt") # .pth  or .pt


# In[47]:


# plot the training losses
import pandas as pd
import matplotlib.pyplot as plt

loss_df = pd.DataFrame({
    "Training Loss": train_losses,
    "Validation Loss": valid_losses
})
plt.plot(loss_df["Training Loss"], label = "Training Loss")
plt.plot(loss_df["Validation Loss"], label = "Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Losses")

plt.legend()


# In[49]:


# Loading the best Model

model.load_state_dict(torch.load("best_model.pt"))


# In[54]:


# Evaluate Our Model

model.eval()
with torch.no_grad():
    train_pred = model(X_train_tensor)
    test_pred = model(X_test_tensor)

    train_mse_loss = criterion(train_pred, y_train_tensor)
    test_mse_loss = criterion(test_pred, y_test_tensor)

print("Training MSE : ", train_mse_loss.item())
print("Testing MSE : ", test_mse_loss.item())
    


# In[55]:


# R2 Score
from sklearn.metrics import r2_score
print( "R2 = ", r2_score(y_test, test_pred))


# In[ ]:




