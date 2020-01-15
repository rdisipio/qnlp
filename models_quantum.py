import numpy as np
import pennylane as qml
import tensorflow as tf
import tensorflow_hub as hub

from functools import partial
from tensorflow import keras
from tensorflow.keras import layers

class VQC(tf.keras.Model):
    def __init__(self, n_categories, n_qubits=4, q_depth=6, **kwargs):
        super(VQC, self).__init__(name="VariationalQuantum")

        self.n_categories = n_categories
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        self.before_quantum = layers.Dense(self.n_qubits, activation='tanh')
        self.class_output = layers.Dense(self.n_categories, activation='softmax')

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        super(VQC, self).__init__(**kwargs)

    def call(self, inputs):
        x = self.before_quantum(inputs)

        x_out = self.class_output(x)

        return x_out