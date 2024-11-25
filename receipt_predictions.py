import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import os

# Read the data
data = pd.read_csv('data.csv', sep=',', comment='#', header=None, names=['Date', 'Receipt_Count'])

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' as index
data.set_index('Date', inplace=True)

# Get receipt counts
receipt_counts = data['Receipt_Count'].values

# Normalize the data to [-1, 1]
min_value = receipt_counts.min()
max_value = receipt_counts.max()

data['Normalized_Count'] = (receipt_counts - min_value) / (max_value - min_value)
data['Normalized_Count'] = data['Normalized_Count'] * 2 - 1  # Scale to [-1,1]

# Define a function to create sequences
def create_sequences(input_data, seq_length):
    inout_seq = []
    L = len(input_data)
    for i in range(L - seq_length):
        seq = input_data[i:i+seq_length]
        label = input_data[i+seq_length]
        inout_seq.append((seq, label))
    return inout_seq

# Create sequences
seq_length = 7
input_data = data['Normalized_Count'].values
inout_seq = create_sequences(input_data, seq_length)

# Define the LSTM model
class ReceiptLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(ReceiptLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Instantiate the model
model = ReceiptLSTM()

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    for seq, labels in inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                             torch.zeros(1,1,model.hidden_layer_size))

        seq = torch.FloatTensor(seq)
        labels = torch.FloatTensor([labels])

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {single_loss.item()}')

# Save the trained model
model_path = 'trained_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'min_value': min_value,
    'max_value': max_value,
    'seq_length': seq_length
}, model_path)

print(f"Model saved to {model_path}")
