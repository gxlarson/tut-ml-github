from pprint import pprint
from transformers import pipeline

clf_zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "how much money is in my bank account"

zs_labels = ['balance', 'income', 'routing number']

output = clf_zs(text, zs_labels)

pprint(output)
