import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

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

    measurements_np = measurements.cpu().numpy()

    # Preprocessing
    scaler = StandardScaler()
    measurements_np = scaler.fit_transform(measurements_np)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    measurements_np = poly.fit_transform(measurements_np)

    measurements_tensor = torch.tensor(measurements_np, dtype=torch.float32).to(device)

    # Initialize the model
    input_size = measurements_tensor.shape[1]
    model = WaistPredictor(input_size).to(device)

    # Load trained weights
    model.load_state_dict(torch.load("models/waist_predictor.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        predicted_waists = model(measurements_tensor)

    return predicted_waists


