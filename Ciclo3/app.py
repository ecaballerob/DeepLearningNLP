import os
import numpy as np
import spacy
import random
import pickle
from flask import Flask, request, render_template

import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('es_core_news_sm')

category_answers = pickle.load(
    open('./data/conversations_category_answers.pkl', 'rb')
    )

vectorized_bow = pickle.load(
    open('./data/conversations_vectorizer_bow.pkl','rb')
    )

categories = pickle.load(
        open('./data/conversations_categories.pkl', 'rb')
        )

model = pickle.load(
    open('./data/conversations_model.pkl','rb')
    )


def text_pre_process(message: str):
    tokens = nlp(message)
    new_tokens = [t.orth_ for t in tokens if not t.is_punct]
    new_tokens = [t.lower() for t in new_tokens]
    clean_message = ' '.join(new_tokens)
    return clean_message

def bow_representation(message: str) -> np.array: # type: ignore
    bow_message = vectorized_bow.transform(
        [message]
        ).toarray()
    return bow_message

def get_prediction(
        bow_message: np.array # type: ignore
        ) -> int:
    prediction = model(bow_message).numpy()
    index = np.argmax(prediction)
    predicted_category = categories[index]
    return predicted_category

def get_answer(category: int) -> str:
    answers = category_answers[category]
    ans = random.choice(answers)
    return ans

app = Flask(__name__, template_folder='./templates')

@app.route('/healtCheck')
def index():
    return "true"



@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/question", methods=['POST'])
def resolve_question():
     conversations = []
     if request.form['question']:
        raw_question = request.form['question']
        clean_question = text_pre_process(raw_question)
        bow_question = bow_representation(clean_question)
        prediction = get_prediction(bow_question)
        bot_answer = get_answer(prediction)
        
        question = 'Usuario: ' + raw_question
        answer = 'ChatBot: ' + bot_answer

        conversations.append(question)
        conversations.append(answer)
        return render_template('response.html', chat=conversations)
     else: 
        return render_template('error.html')

if __name__== '__main__':
    app.run()
