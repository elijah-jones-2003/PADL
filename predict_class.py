import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


def predict_class(images):
    device = torch.device("cuda" if images.is_cuda else "cpu")

    # Initialize the model
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 3)  # output 3 classes
    )
    model = model.to(device)

    # Load the trained weights
    model.load_state_dict(torch.load("fashion_classifier.pth"))
    model.eval()

    # Define transforms for data

    with torch.no_grad():
        # Predict waist circumference
        predicted_class = model(images)

    return predicted_class