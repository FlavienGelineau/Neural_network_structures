from keras.layers import Embedding, Convolution1D, MaxPooling1D, Flatten, Input, Dense, concatenate, Dropout, \
    BatchNormalization
from keras.models import Model
import numpy as np
import random
from sklearn.model_selection import train_test_split

def get_model(show_model_summary = False):
    inputSentence = Input(shape=(50,), name='inputSentence')
    complementInput = Input(shape=(10,), name='complementInput')

    Dense1 = Dense(60, activation='relu')(inputSentence)
    Dense2 = Dense(50, activation='relu')(Dense1)
    Dense3 = Dense(40, activation='relu')(Dense2)

    Dense4 = Dense(40, activation='relu')(complementInput)

    final_output = concatenate([Dense3, Dense4])
    final_output = Dense(30, activation='relu')(final_output)
    final_output = Dense(30, activation='relu')(final_output)
    final_output = Dense(30, activation='relu')(final_output)

    output = Dense(1, activation='relu', name='outputLayer')(final_output)
    network = Model([inputSentence, complementInput], output)

    if show_model_summary:
        network.summary()

    return network

def get_data(len_data, range_of_elts):
    data = np.array([[random.randrange(range_of_elts) for _ in range(59)] for _ in range(len_data)])
    counts = np.array([elt.tolist().count(1) for elt in data]).reshape(-1,1)

    X = np.concatenate((data, counts), axis=1)
    y = np.array([elt1.tolist().count(0) * elt1.tolist().count(1) for elt1 in data])
    return X, y


len_data = 4000
range_of_elts = 3
X, y = get_data(len_data, range_of_elts)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
X_train = [X_train[:, :50], X_train[:, 50:]]
X_val = [X_val[:, :50], X_val[:, 50:]]

network = get_model()

network.compile(loss="mae", optimizer="adam")
network.fit(X_train,
            y_train,
            validation_data=(X_val, y_val),
            verbose=1,
            epochs=600,
            batch_size=50)
