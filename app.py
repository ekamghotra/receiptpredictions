from flask import Flask, render_template
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Define the LSTM model class (same as before)
class ReceiptLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(ReceiptLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq),1,-1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Load the model
model_path = 'trained_model.pth'
checkpoint = torch.load(model_path)
model = ReceiptLSTM()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

min_value = checkpoint['min_value']
max_value = checkpoint['max_value']
seq_length = checkpoint['seq_length']

# Prepare the data for inference
data = pd.read_csv('data.csv', sep=',', comment='#', header=None, names=['Date', 'Receipt_Count'])
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

receipt_counts = data['Receipt_Count'].values

# Normalize the data
data['Normalized_Count'] = (receipt_counts - min_value) / (max_value - min_value)
data['Normalized_Count'] = data['Normalized_Count'] * 2 - 1  # Scale to [-1,1]

input_data = data['Normalized_Count'].values
test_inputs = input_data[-seq_length:].tolist()

# Generate predictions
fut_preds = 365
predictions = []

for i in range(fut_preds):
    seq = torch.FloatTensor(test_inputs[-seq_length:])
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                             torch.zeros(1,1,model.hidden_layer_size))
        pred = model(seq).item()
        test_inputs.append(pred)
        predictions.append(pred)

# Inverse the scaling
predictions = [(pred + 1) / 2 * (max_value - min_value) + min_value for pred in predictions]

# Create date range for 2022
last_date = data.index[-1]
prediction_dates = [last_date + timedelta(days=i+1) for i in range(fut_preds)]

# Create a DataFrame with predictions
future_df = pd.DataFrame({'Date': prediction_dates, 'Predicted_Receipt_Count': predictions})
future_df.set_index('Date', inplace=True)

# Sum up the predictions per month
monthly_predictions = future_df.resample('M').sum()

# Convert index to month names for display
monthly_predictions.index = monthly_predictions.index.strftime('%B %Y')

@app.route('/')
def home():
    # Generate plot
    img = io.BytesIO()
    plt.figure(figsize=(10,5))
    plt.plot(monthly_predictions.index, monthly_predictions['Predicted_Receipt_Count'], marker='o')
    plt.title('Predicted Monthly Receipt Counts for 2022')
    plt.xlabel('Month')
    plt.ylabel('Total Receipts')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Render template with data
    return render_template('index.html',
                           tables=[monthly_predictions.to_html(classes='data', header="true")],
                           plot_url=plot_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
