import torch

class Waist_Predictor():
    pass

waist_predictor = Waist_Predictor()

def predict(measurements):
    waist_predictor.load_state_dict(torch.load("waist_predictor.pth", map_location="cpu"))
    waist_predictor.eval()
    with torch.no_grad():
        predictions = waist_predictor(measurements)
    return predictions