import os
import numpy as np
import spacy
import pandas as pd
import seaborn as sns
import json
import itertools
from sklearn.feature_extraction.text import CountVectorizer
import pickle

import warnings

warnings.filterwarnings('ignore')

#Modelo de Lenguaje
nlp = spacy.load('es_core_news_sm')


with open('../data/conversations.json') as f:
    conversations = json.load(f)


tags = []

for script in conversations:
    message = script['tag']
    tags.append(message)

tags.sort()

with open('tags.pkl', 'wb') as file:
    pickle.dump(tags, file)


patterns = []

for script in conversations:
    message = script['patterns']
    patterns.append(message)

documents = list(itertools.chain.from_iterable(patterns))

patterns_tokens = []

for doc in documents:
    doc_tokens = nlp(doc)
    new_doc_tokens = [t.orth_.lower() for t in doc_tokens if not t.is_stop | t.is_punct]
    patterns_tokens.append(new_doc_tokens)

patterns_tokens.sort()
patterns_tokens_clean = list(
    itertools.chain.from_iterable(patterns_tokens)
    )

with open('vocabulario.pkl', 'wb') as file:
    pickle.dump(patterns_tokens_clean, file)



word_count = {}

for palabra in patterns_tokens_clean:
    if palabra in word_count.keys():
        word_count[palabra] += 1
    else:
        word_count[palabra] = 1

df = pd.DataFrame(
    {'palabra': word_count.keys(),
     'frecuencia': word_count.values()}
    )

df.sort_values(['frecuencia'], ascending=False, inplace=True)

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(documents)

bow = pd.DataFrame(
    x.toarray(),
    columns=vectorizer.get_feature_names_out()
    )
bow.to_csv('bow_amira_patterns.csv', index=False)




