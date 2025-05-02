import torch

class Encoder():
    pass

class Decoder():
    pass

encoder = Encoder()
decoder = Decoder()

def encode(images):
    encoder.load_state_dict(torch.load("encoder.pth", map_location="cpu"))
    encoder.eval()
    images = images.clamp(0, 1)  # ensure proper input
    with torch.no_grad():
        latents = encoder(images)
    return latents

def decode(latents):
    decoder.load_state_dict(torch.load("decoder.pth", map_location="cpu"))
    decoder.eval()
    with torch.no_grad():
        recon = decoder(latents)
    return recon