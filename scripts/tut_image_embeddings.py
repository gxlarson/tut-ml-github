import torch
import clip
import numpy as np
from PIL import Image

def cosine_similarity(u, v):
    a = u.ravel()
    b = v.ravel()
    sim = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
    return sim

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

im1 = preprocess(Image.open("im1.png")).unsqueeze(0).to(device)
im2 = preprocess(Image.open("im2.png")).unsqueeze(0).to(device)

im1_embedding = model.encode_image(im1).detach().numpy()
im2_embedding = model.encode_image(im2).detach().numpy()

sim = cosine_similarity(im1_embedding, im2_embedding)
print(sim)
