# fake_news_detection
In today’s digital age, misinformation spreads rapidly across social media and news platforms, causing confusion and sometimes serious consequences. To combat this, I have developed a Fake News Detector—a machine learning-powered web application that classifies news articles as either Real or Fake based on their content.
Project Overview
This project uses natural language processing (NLP) techniques combined with machine learning algorithms to analyze the text of news articles and determine their authenticity. The model is trained on a carefully curated dataset of Indian news articles containing both verified real news and known fake news.

The core pipeline involves:

Data Preprocessing: Cleaning the news text by removing punctuation, stopwords, and irrelevant content.

Vectorization: Converting text into numerical data using TF-IDF vectorizer.

Model Training: Training a classifier (such as Logistic Regression) to distinguish between real and fake news.

Evaluation: Assessing the model performance with metrics like accuracy, precision, recall, and F1-score, which are around 99%, showing strong reliability.

Web Application
The project includes a Flask-based web app that allows users to input news text and get instant predictions. The app shows whether the news is likely Real or Fake and also provides a confidence score indicating how sure the model is about its prediction. The interface is simple and intuitive, making it accessible for anyone to use.

Why This Project Matters
Fake news can influence public opinion, elections, and even public health. Tools like this can help people critically evaluate the news they consume and reduce the spread of misinformation. Although automated detection is not perfect, this model provides a useful first step in verifying news content.

Technologies Used
Python (pandas, scikit-learn, nltk)

Flask for the web interface

HTML/CSS for frontend styling

Pickle for saving/loading the trained model

How to Run
Train the model using train_model.py

Start the Flask app using app.py

Open the browser at http://127.0.0.1:5000/ and test news articles
