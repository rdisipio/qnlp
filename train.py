#!/usr/bin/env python

import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from models import *

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

#tf.compat.v1.enable_eager_execution()

#########################################

if __name__ == '__main__':

    LATENT_DIM = 16
    TEST_SIZE = 0.2
    N_EPOCHS = 20
    BATCH_SIZE = 128

    f_name = "arxiv_abstracts.pkl"
    f = open(f_name, 'rb')
    df = pickle.load(f)

    categories = list(set(df['category_txt'].values))
    n_categories = len(categories)
    print("There are {} known categories: {}".format(n_categories, categories))

    X_txt = df['abstract'].values

    y = np.array(df['category_id'].values)
    y = to_categorical(y)

    X_txt_train, X_txt_test, y_train, y_test = train_test_split(X_txt, y)

    print("Training set has {} samples".format(X_txt_train.shape[0]))
    print("Testing set has {} samples".format(X_txt_test.shape[0]))

    # initialize USE embedder
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(module_url)

    #model = make_model(embed, n_categories=n_categories, latent_dim=LATENT_DIM)
    model = make_model_quantum(embed, n_categories=n_categories)

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        loss='categorical_crossentropy',
        optmizer=optimizer,
        metrics=['acc'],
    )

    print("Training...")

    callback = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.005)

    model.fit(
        X_txt_train, y_train,
        epochs=N_EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[],
    )
    print("Done training")

    print("Testing...")
    test_score = model.evaluate(X_txt_test, y_test, verbose=2)

