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

with open('../data/sample.json') as f:
    conversations = json.load(f)

#pprint(conversations)

category_answers = {}

for conversation in conversations:
    cat = conversation['tag']
    category_answers[cat] = conversation['responses']

#TODO hacer pickle

pprint(category_answers)

nlp = spacy.load('es_core_news_sm')
questions = []

for script in conversations:
    question = script['patterns']
    questions.append(question)

documents = list(itertools.chain.from_iterable(questions))

temp = pd.DataFrame(conversations).explode(['patterns'])

pprint(temp)

pprint(documents)

question_processed = []

for doc in documents:
    tokens = nlp(doc)
    new_tokens = [t.orth_ for t in tokens if not t.is_punct]
    new_tokens = [t.lower() for t in new_tokens]
    question_processed.append(' '.join(new_tokens))

pprint(question_processed)

df_conversation = pd.DataFrame(conversations).explode(
        ['patterns']
        ).reset_index(drop=True)

pprint(df_conversation[['patterns', 'tag']])

x = vectorizer.fit_transform(question_processed)

bow = pd.DataFrame(
    x.toarray(),
    columns=vectorizer.get_feature_names_out()
    )

#TODO crear pickle de bow

pprint(bow.sample(10))

processed_data = pd.concat(
    [bow,
     df_conversation[['tag']]
     ], axis=1)


processed_data = processed_data.sample(
    frac=1,
    random_state=123
).reset_index(drop=True)

processed_data.info()

dummies = pd.get_dummies(processed_data['tag'], dtype=int)
pprint(dummies)

sample_categories = list(pd.get_dummies(processed_data['tag']).columns)
pprint(sample_categories)

#TODO hacer piclke

dim_x = len(processed_data._get_numeric_data().to_numpy()[0])
dim_y = len(pd.get_dummies(processed_data['tag'], dtype=int).to_numpy()[0])

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
    verbose=1
    )
