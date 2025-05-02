import torch

class FashionClassifier():
    pass

fashion_classifier = FashionClassifier()

def predict_class(images):
    fashion_classifier.load_state_dict(torch.load("waist_predictor.pth", map_location="cpu"))
    fashion_classifier.eval()
    with torch.no_grad():
        predictions = fashion_classifier(images)
    return predictions