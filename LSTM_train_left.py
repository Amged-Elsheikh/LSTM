# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import os
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import seaborn as sns

# %% [markdown]
# ### Data Processing 

# %%
# Import data from training & testing dataset
training_csv_file = 'dataset_2020_training_left.csv'
testing_csv_file = 'dataset_2020_testing1_left.csv'
training_set = pd.read_csv(training_csv_file, header=0)
testing_set = pd.read_csv(testing_csv_file, header=0)

#The output is: knee/ankle angle/torque l/r
out_num = 4
input_size = 49
#Get the label
Label_df = pd.read_csv(training_csv_file, header=0, nrows = 1)
Label= Label_df.columns.values
output_label = np.array(Label[-out_num:])

#Get the values and convert to float type
training_values = training_set.values
testing_values = testing_set.values
time_train = training_values[:,0]
time_test  = testing_values[:,0]
training_data = training_values[:,1:].astype('float32')
testing_data = testing_values[:,1:].astype('float32')


# %%
def get_scaler(ndarray):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(ndarray)
    return scaler


# %%
#Get scaler for inputs/outputs
X_scaler_training = get_scaler(training_data[:,:input_size])
Y_scaler_training = get_scaler(training_data[:,-out_num:])
X_scaler_testing = get_scaler(testing_data[:,:input_size])
Y_scaler_testing = get_scaler(testing_data[:,-out_num:])


X_train = X_scaler_training.transform(training_data[:,:input_size])
Y_train = Y_scaler_training.transform(training_data[:,-out_num:])
X_test = X_scaler_testing.transform(testing_data[:,:input_size])
Y_test = Y_scaler_testing.transform(testing_data[:,-out_num:])

#Get scaler for the whole training data
train_data_scaler = get_scaler(training_data)
test_data_scaler = get_scaler(testing_data)

#Transform training data into scaled data and tensor data type
train_data_normalized = train_data_scaler.transform(training_data)
test_data_normalized =test_data_scaler.transform(testing_data)


# %%
def reshape(A, timesteps):
    return A.reshape(int(A.shape[0] / timesteps), timesteps, A.shape[1])

#Return tupple containing inputs and labels
def create_inout_sequences(input_data, size, out_num, time_steps):
    sequence = []
    outs =  []
    L = len(input_data)
    for i in range(L-time_steps):
        seq = input_data[i:i+time_steps,:size]
        label = input_data[i:i+time_steps,-out_num:]
        sequence.append(seq)
        outs.append(label)
    sequence = np.array(sequence)
    outs = np.array(outs)
    return sequence,outs


# %%
train_input, train_output = create_inout_sequences(train_data_normalized, input_size, out_num, 50)
test_input, test_output = create_inout_sequences(test_data_normalized, input_size, out_num, 50)


# %%
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(train_input), torch.from_numpy(train_output))
val_data = TensorDataset(torch.from_numpy(test_input), torch.from_numpy(test_output))
test_data = TensorDataset(torch.from_numpy(test_input), torch.from_numpy(test_output))

batch_size = 64

train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=1)


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %% [markdown]
# ### Model 

# %%

class Model(nn.Module):

    def __init__(self, input_dim = input_size, 
                 hidden_dim = 256, 
                 output_dim=out_num,
                 num_layers=3,
                 drop_prob=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, 
                            num_layers, dropout=drop_prob, batch_first=True)
        
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.reshape(-1, self.hidden_dim)
        out = self.linear(lstm_out)
        
        out = out.view(-1, out.size(-1))    
        # Only take the output from the final timetep
        return out, hidden

# %% [markdown]
# ### Training Model

# %%
lr=0.001
model = Model()
model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 200

best_val_loss = 1000
for epoch in range(num_epochs):
    train_loss_epoch = 0.0
    
    # train
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        # zero grad
        model.zero_grad()
        
        # forward
        Y = Y.view(-1, out_num)
        Y_pred, _ = model(X)
        
        # compute loss
        loss_batch = loss_fn(Y, Y_pred)
        train_loss_epoch += loss_batch * Y_pred.size(0)
        
        # backward
        loss_batch.backward()

        # optimize
        optimizer.step()
    
    val_loss_epoch = 0.0
    # validation
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        
        # forward
        Y = Y.view(-1, out_num)
        Y_pred, _ = model(X)

        # compute loss
        loss_batch = loss_fn(Y, Y_pred)
        val_loss_epoch += loss_batch * Y_pred.size(0)
    
    time_steps = 50
    train_loss_epoch = train_loss_epoch / (len(train_data)*time_steps)
    val_loss_epoch = val_loss_epoch / (len(test_data)*time_steps)
    print("Epoch: %2d, train loss: %.4f, val loss: %.4f" % (epoch, 
                                                            train_loss_epoch.item(),
                                                            val_loss_epoch.item()))
    
    # save model
    if val_loss_epoch < best_val_loss:
        print("Save best model at epoch %2d" % epoch)
        torch.save(model.state_dict(), "best_model_server_left.pth")
        best_val_loss = val_loss_epoch
