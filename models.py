import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers

import pennylane as qml

# from models_quantum import VQC

from functools import partial

import numpy as np

def USELayer(embed,x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))

def make_model(embed, n_categories, latent_dim=16, embedding_dim=512):
    UniversalEmbedding = partial(USELayer, embed)

    text_in = keras.Input( shape=(1,), dtype=tf.string, name="text_in")

    x = layers.Lambda(UniversalEmbedding, output_shape=(embedding_dim, ))(text_in)

    x = layers.Dense(latent_dim, activation='relu')(x)

    x_out = layers.Dense(n_categories, activation='softmax')(x)

    return keras.Model(inputs=text_in, outputs=x_out, name="ClassicalPreprintClassifier")


def make_model_classical(n_categories, latent_dim=16, embedding_dim=512):
    
    text_in = keras.Input( shape=(embedding_dim,), dtype=tf.float64, name='text_in') # (None, 512)

    x = layers.Dense(latent_dim, activation='tanh', dtype=tf.float64)(text_in)

    x_out = layers.Dense(n_categories, activation='softmax')(x)

    return keras.Model(inputs=text_in, outputs=x_out, name="ClassicalPreprintClassifier")
