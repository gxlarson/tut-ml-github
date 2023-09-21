from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

s1 = "what is my bank balance"

e1 = model.encode([s1])[0]


