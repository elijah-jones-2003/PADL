import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.LayerNorm(size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class WaistPredictor(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super().__init__()
        hidden_size = 64

        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.res_block1 = ResidualBlock(hidden_size, dropout=dropout)
        self.res_block2 = ResidualBlock(hidden_size, dropout=dropout)
        self.res_block3 = ResidualBlock(hidden_size, dropout=dropout)

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attention_weights = self.attention(x)
        x = x * attention_weights
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.output_layer(x)
        return x

def predict(measurements):
    device = torch.device("cuda" if measurements.is_cuda else "cpu")

    # Preprocessing

    # Initialize the model
    model = WaistPredictor()
    model = model.to(device)

    # Load the trained weights
    model.load_state_dict(torch.load("waist_predictor.pth"))
    model.eval()

    with torch.no_grad():
        # Predict waist circumference
        predicted_waists = model(measurements.float())

    return predicted_waists