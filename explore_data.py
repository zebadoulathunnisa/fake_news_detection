import pickle
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

test_sentences = [
    "The government has announced new policies for education.",
    "Scientists found that chocolate cures cancer overnight."
]

for sentence in test_sentences:
    processed = preprocess(sentence)
    vect = vectorizer.transform([processed])
    pred = model.predict(vect)[0]
    label = "REAL" if pred == 0 else "FAKE"
    print(f"Sentence: {sentence}\nPrediction: {label}\n")
