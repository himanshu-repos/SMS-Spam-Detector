import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_data(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        # Only considering if i is Alphabet or a Number
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl' ,'rb'))
model = pickle.load(open('model.pkl' ,'rb'))

st.title("Email/SMS Classifier")

input_sms = st.text_area("Enter the Message")

if(st.button('Detect')):
    transformed_sms = transform_data(input_sms)  # Data Preprocessing
    vectorized_sms = tfidf.transform([transformed_sms])     # Text Vectorizer
    result_sms = model.predict(vectorized_sms)[0]
    if result_sms == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# myenv\Scripts\activate