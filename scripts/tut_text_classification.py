import json
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

with open('clinc150.json', 'r') as f:
    data = json.load(f)
labels = ['balance', 'income', 'routing']

def predict(text, vectorizer, model):
    features = vectorizer.transform([text])
    output = model.predict_proba(features)
    return output

def subset(split, labels):
    keeps = []
    for (sample, label) in split:
        if label in labels:
            # keep it
            keeps.append((sample, label))
    return keeps

train = subset(data['train'], labels)
test = subset(data['test'], labels)

feature_extractor = CountVectorizer()
x_train = feature_extractor.fit_transform([text for (text, _) in train])
x_test = feature_extractor.transform([text for (text, _) in test])
y_train_labels = [label for (_, label) in train]
y_test_labels = [label for (_, label) in test]

label_map = {'balance': 0, 'income': 1, 'routing': 2}
y_train = [label_map[label] for label in y_train_labels]
y_test = [label_map[label] for label in y_test_labels]

clf = svm.SVC(probability=True)
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

print(predict("what is my balance please", feature_extractor, clf))

print(predict("a random example here", feature_extractor, clf))


print('----- now training embedding model -----')

from sentence_transformers import SentenceTransformer

feature_extractor = SentenceTransformer('all-MiniLM-L6-v2')

x_train_emb = feature_extractor.encode([x for (x, _) in train])

clf_emb = svm.SVC()
clf_emb.fit(x_train_emb, y_train)
