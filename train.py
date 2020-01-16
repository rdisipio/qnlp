#!/usr/bin/env python

import pickle
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from models_quantum import make_model_quantum
from models import make_model_classical

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping

#tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx('float64') 

# initialize USE embedder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)

def embed_text(X_txt):
    print("Embedding input text...")

    X = np.array([np.array(x) for x in embed(X_txt)])

    return X

#########################################

if __name__ == '__main__':

    LATENT_DIM = 16
    TEST_SIZE = 0.2
    N_EPOCHS = 20
    BATCH_SIZE = 32
    N_SAMPLES = 1000
    CLASSIFIER = 'quantum'

    if len(sys.argv) > 1:
        CLASSIFIER = sys.argv[1]

    f_name = "arxiv_abstracts.pkl"
    f = open(f_name, 'rb')
    df = pickle.load(f)

    df = df.sample(N_SAMPLES)

    categories = list(set(df['category_txt'].values))
    n_categories = len(categories)
    print("There are {} known categories: {}".format(n_categories, categories))

    X_txt = [x for x in df['abstract'].values]

    X = embed_text(X_txt)
    embedding_dim = X.shape[-1]
    print("Embeddings shape: {}".format(X.shape))

    y = np.array(df['category_id'].values)
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("Training set has {} samples".format(X_train.shape[0]))
    print("Testing set has {} samples".format(X_test.shape[0]))

    if CLASSIFIER == 'quantum':
        model = make_model_quantum(n_categories=n_categories, 
                                n_qubits=4, 
                                n_layers=1, 
                                embedding_dim=embedding_dim)
    elif CLASSIFIER == 'classical':
        model = make_model_classical(n_categories=n_categories, 
                                    latent_dim=LATENT_DIM, 
                                    embedding_dim=embedding_dim)
    else:
        raise Exception("Unknown method of classification: {}".format(CLASSIFIER))
    
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        loss='categorical_crossentropy',
        optmizer=optimizer,
        metrics=['acc'],
    )

    # print("Trainable variables:")
    # print(model.trainable_variables)
    
    print("Training...")

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.01)
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks = [early_stopping_callback]

    model.fit(
        X_train, y_train,
        epochs=N_EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
    )
    print("Done training")

    print("Testing...")
    test_score = model.evaluate(X_test, y_test, verbose=2)

