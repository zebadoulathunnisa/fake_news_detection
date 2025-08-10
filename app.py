from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Basic preprocessing: lowercasing and removing stopwords
    text = text.lower()
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    if request.method == 'POST':
        news_text = request.form['news_text']
        processed_text = preprocess(news_text)
        vect_text = vectorizer.transform([processed_text])
        prediction = model.predict(vect_text)[0]
        confidence = max(model.predict_proba(vect_text)[0])

        if confidence >= 0.75:
            if prediction == 1:
                prediction_text = "✅ This news seems to be REAL."
            else:
                prediction_text = "❌ This news seems to be FAKE."
        else:
            prediction_text = "❌ This news seems to be FAKE."

        return render_template('index.html', prediction_text=prediction_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
