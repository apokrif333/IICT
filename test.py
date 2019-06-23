import pandas as pd
import numpy as np
import re
import functools
import operator
from tqdm import tqdm

df = pd.read_csv('data/lenta-ru-news.csv')

clear_df = pd.DataFrame({})
clear_df['text'] = df.text.str.lower().str.replace(r"(\n|\t|\xa0)", ' ').str.replace(r"\.\s+", r".").str.replace(
    r"(«|»|,|:|\(|\)|\[|\]|,)", "")

clear_df['text'] = clear_df['text'].str.split(".")
clear_df.dropna(inplace=True)
all_sentences = functools.reduce(operator.iconcat, clear_df['text'], [])
true_idx = [idx for idx, sent in enumerate(all_sentences) if sent is not '']
all_sentences = list(operator.itemgetter(*true_idx)(all_sentences))

need_words = ["прокурор", "следствие", "авария", "гражданин", "указ", "акция", "белка", "граф", "орган", "вид"]
need_sentences = []
for idx in tqdm(range(len(all_sentences))):
    if len(set(need_words) & set(all_sentences[idx].split())) != 0:
        need_sentences.append(all_sentences[idx])

words_dict = dict()

import tensorflow as tf
import tensorflow_hub as hub

start = 31_000
step = 1_000
for sent_idx in range(start, len(need_sentences), step):
    print(sent_idx)

    elmo_tf = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
                         trainable=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    sent_batch = need_sentences[sent_idx - step:sent_idx]
    vectors = sess.run(elmo_tf(sent_batch, signature="default", as_dict=True)["word_emb"])
    print(vectors.shape)

    for sentence in sent_batch:
        new_keys = set(sentence.split()) - set(words_dict.keys())
        words_dict.update(
            dict(zip(new_keys, len(new_keys) * [[]]))
        )

    for cur_sentence, cur_vector in tqdm(zip(sent_batch, vectors)):
        split_sentence = cur_sentence.split()
        for word in range(len(split_sentence)):
            words_dict[split_sentence[word]].append(cur_vector[word])

    for word in need_words:
        difference = []
        for i in range(len(words_dict[word]) - 1):
            difference.append(np.linalg.norm(words_dict[word][i] - words_dict[word][i + 1]))

        print(word, np.std(difference))

    sess.close()
