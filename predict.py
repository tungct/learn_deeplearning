from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import re
import numpy as np
from pickle import dump
from pickle import load
from keras.models import load_model
# load the tokenizer
tokenizer = load(open('test.pkl', 'rb'))
vacab = []
def prepare_sentence(seq, maxlen):
    # Pads seq and slides windows
    x = []
    y = []
    for i, w in enumerate(seq):
        x_padded = pad_sequences([seq[:i]],
                                 maxlen=maxlen - 1,
                                 padding='pre')[0]  # Pads before each sequence

        x.append(x_padded)
        y.append(w)
    return x, y

def dataset_preparation(data):
    # basic cleanup
    corpus = data.lower().split("#")

    # tokenization
    tokenizer.fit_on_texts(corpus)
    vocab = tokenizer.word_index
    total_words = len(tokenizer.word_index) + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words, vocab

def get_vocab(data):
    # basic cleanup
    corpus = data.lower().split("#")

    # tokenization
    tokenizer.fit_on_texts(corpus)
    vocab = tokenizer.word_index
    total_words = len(tokenizer.word_index) + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])

    return max_sequence_len, total_words, vocab

def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

def score_sentence(sentence, maxlen, vocab):
    tok = tokenizer.texts_to_sequences([sentence])[0]
    x_test, y_test = prepare_sentence(tok, maxlen)
    x_test = np.array(x_test)
    y_test = np.array(y_test) - 1  # The word <PAD> does not have a class
    p_pred = model.predict(x_test)
    vocab_inv = {v: k for k, v in vocab.items()}
    log_p_sentence = 0
    for i, prob in enumerate(p_pred):
        word = vocab_inv[y_test[i] + 1]  # Index 0 from vocab is reserved to <PAD>
        history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
        prob_word = prob[y_test[i]]
        log_p_sentence += np.log(prob_word)
        print('P(w={}|h={})={}'.format(word, history, prob_word))
    print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))

data = open('data.txt').read()

max_sequence_len, total_words, vocab = get_vocab(data)
model = load_model('test.h5')
# print(generate_text("em của", 3, max_sequence_len))

sc1 = score_sentence("một trong số các khuyết điểm của tôi là yêu em", max_sequence_len, vocab)
sc2 = score_sentence("một trọng số các khuyết điểm của tôi là yêu em", max_sequence_len, vocab)


