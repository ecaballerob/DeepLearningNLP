import os
import numpy as np
import spacy
import pandas as pd
import seaborn as sns
import json
import random
import pickle
from pprint import pprint
import itertools
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import warnings

warnings.filterwarnings('ignore')
vectorizer = CountVectorizer()
nlp = spacy.load('es_core_news_sm')
category_answers = {}

def text_pre_process(message: str):
    tokens = nlp(message)
    new_tokens = [t.orth_ for t in tokens if not t.is_punct]
    new_tokens = [t.lower() for t in new_tokens]
    clean_message = ' '.join(new_tokens)
    return clean_message

def bow_representation(message: str)-> np.array:
    bow_message = vectorizer.transform(
        [message]
        ).toarray()
    return bow_message

def get_prediction(
        bow_message: np.array
        )-> int:
    prediction = model(bow_message).numpy()
    index = np.argmax(prediction)
    predicted_category = sample_categories[index]
    return predicted_category

def get_answer(category: str)-> str:
    answers = category_answers[category]
    ans = random.choice(answers)
    return ans

def main():
    with open('../data/conversations.json') as f:
        conversations = json.load(f)


    for conversation in conversations:
        cat = conversation['tag']
        category_answers[cat] = conversation['responses']

    with open('conversations_category_answers.json', 'w') as file:
        json.dump(category_answers, file)

    questions = []

    for script in conversations:
        question = script['patterns']
        questions.append(question)

    documents = list(itertools.chain.from_iterable(questions))

    pd.DataFrame(conversations).explode(['patterns'])

    question_processed = []

# se usa funcion pre process para evitar duplicidad de codigo
    for doc in documents:
        question_processed.append(text_pre_process(doc))

    df_conversation = pd.DataFrame(conversations).explode(
            ['patterns']
            ).reset_index(drop=True)

    x = vectorizer.fit_transform(question_processed)

    bow = pd.DataFrame(
        x.toarray(),
        columns=vectorizer.get_feature_names_out()
        )

    pickle.dump(
        vectorizer,
        open('conversations_vectorizer_bow.pkl', 'wb')
        )

    processed_data = pd.concat(
        [bow,
         df_conversation[['tag']]
         ], axis=1)


    processed_data = processed_data.sample(
        frac=1,
        random_state=123
    ).reset_index(drop=True)

    pd.get_dummies(processed_data['tag'], dtype=int)
    
    global sample_categories
    sample_categories = list(pd.get_dummies(processed_data['tag']).columns)

    pickle.dump(
            sample_categories,
            open('conversations_categories.pkl', 'wb')
            )

    dim_x = len(processed_data._get_numeric_data().to_numpy()[0])
    dim_y = len(pd.get_dummies(processed_data['tag'], dtype=int).to_numpy()[0])

    global model
    model = Sequential([
        Dense(25, input_shape=(dim_x,), activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dropout(0.2),
        Dense(dim_y, activation='softmax')
        ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
        )

    X_train = processed_data._get_numeric_data().to_numpy()

    Y_train = pd.get_dummies(processed_data['tag'], dtype=float).to_numpy()

    hist = model.fit(
        X_train,
        Y_train,
        epochs=500,
        batch_size=100,
        verbose=0
        )

if __name__ == "__main__":
    main()
    new_message = input("Escribe tu pregunta: ")
    bot_answer = get_answer(
            get_prediction(
                bow_representation(
                    text_pre_process(new_message)
                    )
                )
            )
    print("Usuario:", new_message)
    print("Chatbot:", bot_answer)
