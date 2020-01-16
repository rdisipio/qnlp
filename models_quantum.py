import numpy as np

import pennylane as qml

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from functools import partial
from models import USELayer

class VariationalQuantumCircuit(layers.Layer):
    def __init__(self, 
            n_categories, 
            n_qubits=4, 
            n_layers=6,
            device="default.qubit",
            **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        if "dynamic" in kwargs:
            del kwargs["dynamic"]

        super(VariationalQuantumCircuit, self).__init__(dynamic=True)

        self.n_categories = n_categories
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device

        self.before_quantum = layers.Dense(self.n_qubits, activation='tanh')
        self.class_output = layers.Dense(self.n_categories, activation='softmax')

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        def _circuit(inputs, parameters):
            qml.templates.embeddings.AngleEmbedding(inputs, wires=list(range(self.n_qubits)))
            qml.templates.layers.StronglyEntanglingLayers(parameters, wires=list(range(self.n_qubits)))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.dev = qml.device(self.device, wires=self.n_qubits)
        self.layer = qml.QNode(_circuit, self.dev, interface="tf")

    def apply_layer(self, *args):
        return tf.keras.backend.cast_to_floatx(self.layer(*args))
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        assert input_dim == self.n_qubits

        self.rotations = self.add_weight(
            shape=(self.n_layers, input_dim, 3),
            name="rotations",
            initializer="random_uniform",
            regularizer=None,
            constraint=None
            )
        self.built = True
    
    def call(self, inputs):
        return tf.stack([self.apply_layer(i, self.rotations) for i in inputs])
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {
            "units": self.units,
            "device": self.device,
            "n_layers": self.n_layers,
        }
        base_config = super(VariationalQuantumCircuit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def make_model_quantum(embed, n_categories, n_qubits=4, n_layers=2):
    
    UniversalEmbedding = partial(USELayer, embed)

    text_in = keras.Input( shape=(1,), dtype=tf.string, name="text_in")

    x = layers.Lambda(UniversalEmbedding, name="USE_embedding", dtype=tf.float64)(text_in)
    x = layers.Dense(n_qubits, activation='tanh', dtype=tf.float64)(x)

    x = VariationalQuantumCircuit(
            n_categories=n_categories, 
            n_qubits=n_qubits, 
            n_layers=n_layers)(x)
    
    assert x.shape[-1] == n_qubits

    x_out = layers.Dense(n_categories, activation='softmax', dtype=tf.float64)(x)
    #x = layers.Dense(2)(x)
    #x_out = layers.Activation("softmax")(x)

    return keras.Model(inputs=text_in, outputs=x_out, name="QuantumPreprintClassifier")