import clip
import torch
import numpy as np
from PIL import Image

def softmax(z):
    numerator = np.exp(z)
    denominator = np.sum(np.exp(z))
    probs = numerator / denominator
    return probs

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

im = preprocess(Image.open("im1.png")).unsqueeze(0).to(device)
classes = ["a dog", "a cat", "a barcode"]
text = clip.tokenize(classes).to(device)

im_logits, text_logits = model(im, text)
im_logits = im_logits.detach().numpy()[0]

scores = softmax(im_logits)

for i, label in enumerate(classes):
    print(f'{label}: ' + str(scores[i]))
