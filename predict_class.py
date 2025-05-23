import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms


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
    model.load_state_dict(torch.load("models/fashion_classifier.pth", map_location=torch.device('cpu')))
    model.eval()

    # Define transforms for data
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    normalized_images = transform(images)
    normalized_images = normalized_images.to(device)

    with torch.no_grad():
        # Predict waist circumference
        predicted_class = model(normalized_images)

    return predicted_class