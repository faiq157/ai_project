import streamlit as st
import pandas as pd
from datasets import load_dataset
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import joblib

# Load the dataset
dataset = load_dataset("emotion")
df = pd.DataFrame(dataset['train'])

# Text preprocessing function
nltk.download('stopwords')
nltk.download('punkt')
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

# Preprocess and vectorize the text data
df['processed_text'] = df['text'].apply(preprocess_text)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['label']

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model and vectorizer
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Load the model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
st.title("Emotion Detection from Text")
user_input = st.text_area("Enter your text here:")
if st.button("Analyze"):
    processed_input = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([processed_input])
    prediction = model.predict(vectorized_input)
    st.write(f"Detected Emotion: {prediction[0]}")
