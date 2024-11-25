import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
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

    if epoch%10 == 0:
        print(f'Epoch {epoch} Loss: {single_loss.item()}')

# Generate predictions for 2022
fut_preds = 365
test_inputs = input_data[-seq_length:].tolist()

model.eval()

for i in range(fut_preds):
    seq = torch.FloatTensor(test_inputs[-seq_length:])
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                             torch.zeros(1,1,model.hidden_layer_size))
        test_inputs.append(model(seq).item())

# Inverse the scaling
predictions = test_inputs[seq_length:]
predictions = [(pred + 1) / 2 * (max_value - min_value) + min_value for pred in predictions]

# Create date range for 2022
last_date = data.index[-1]
prediction_dates = [last_date + timedelta(days=i+1) for i in range(fut_preds)]

# Create a DataFrame with predictions
future_df = pd.DataFrame({'Date': prediction_dates, 'Predicted_Receipt_Count': predictions})
future_df.set_index('Date', inplace=True)

# Sum up the predictions per month
monthly_predictions = future_df.resample('M').sum()

print("Predicted receipt counts for each month in 2022:")
print(monthly_predictions)
