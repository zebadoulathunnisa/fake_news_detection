import pandas as pd
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import nltk

# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("data/train.csv")
df = df.fillna('')

# Show columns and labels for debugging
print("Columns:", df.columns)
print("Unique label values in 'class' column:")
print(df['class'].unique())
print("Label counts:")
print(df['class'].value_counts())

# Keep only Fake and Real news rows, remove bad rows
df = df[df['class'].isin(['Fake', 'Real'])]
df = df.reset_index(drop=True)
print("Cleaned label counts:\n", df['class'].value_counts())

# Combine title and text into single content column
df['content'] = df['title'] + " " + df['text']

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['content'] = df['content'].apply(preprocess)

X = df['content']
y = df['class']

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Train logistic regression model with balanced classes
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
