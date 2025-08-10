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
    ("The Eiffel Tower is located in Paris, France.", 0),  # Real
    ("The Eiffel Tower is located in New York City.", 1)   # Fake
]

for text, expected_label in test_sentences:
    processed = preprocess(text)
    vect = vectorizer.transform([processed])
    pred = model.predict(vect)[0]
    print(f"Input: {text}")
    print(f"Processed: {processed}")
    print(f"Predicted label: {pred} (Expected: {expected_label})")
    print(f"Prediction is {'CORRECT' if pred == expected_label else 'INCORRECT'}\n")
