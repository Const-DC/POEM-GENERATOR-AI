import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import os

# Download and load the text
filepath = tf.keras.utils.get_file(
    'shakespeare.txt', 
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()


characters = sorted(set(text))
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 3


sentences = []
next_chars = []
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_chars.append(text[i + SEQ_LENGTH])


x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

model_path = "text_generator.keras"

# Build and train the model if not already trained
if not os.path.exists(model_path):
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
    model.add(Dense(len(characters)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
    
    model.fit(x, y, epochs=30 , batch_size=256)
    model.save(model_path)
    print("The Model is Trained and Saved :)")
else:
    print("Model already exists :) Delete 'text_generator.keras' to retrain.")

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Sampling function to pick next char
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(np.clip(preds, 1e-10, 1.0)) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def text_generate(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated = sentence

    for _ in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated


for temp in [0.1, 0.4, 0.6, 0.8, 1.0]:
    print(f"\n--- Temperature: {temp} ---")
    print(text_generate(300, temp))
