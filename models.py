import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers

import pennylane as qml

from functools import partial

import numpy as np

def USELayer(embed,x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))

def make_model(embed, n_categories, latent_dim=16, embedding_dim=512):
    UniversalEmbedding = partial(USELayer,embed)

    text_in = keras.Input( shape=(1,), dtype=tf.string, name="text_in")

    x = layers.Lambda(UniversalEmbedding, output_shape=(embedding_dim, ))(text_in)

    x = layers.Dense(latent_dim, activation='relu')(x)

    x_out = layers.Dense(n_categories, activation='softmax')(x)

    return keras.Model(inputs=text_in, outputs=x_out, name="AbstractClassifier")

#####################################

def H_layer(n_qubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(n_qubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, n_qubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])

    for i in range(1, n_qubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class dressed_quantum_circuit(layers.Layer):

    def __init__(self, output_dim, n_qubits=4, q_depth=6, **kwargs):

        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        super(dressed_quantum_circuit, self).__init__(**kwargs)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.q_depth, self.n_qubits),
                                      initializer='uniform',
                                      trainable=True)
        super(dressed_quantum_circuit, self).build(input_shape)  # Be sure to call this at the end

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

    def variational_quantum_circuit(self, q_in):

        q_weights = self.kernel.numpy()

        q_depth = q_weights.shape[0]
        n_qubits = q_weights.shape[1]

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        H_layer(n_qubits)

        # Embed features in the quantum node
        RY_layer(q_in)

        # Sequence of trainable variational layers
        for k in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[k])

        # Expectation values in the Z basis
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
        return exp_vals
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #@tf.function
    def run_quantum_circuit(self, x):
        q_out = []

        # Apply the quantum circuit to each element of the batch 
        for q_in in x.numpy(): 
            q = qml.QNode(self.variational_quantum_circuit(q_in), self.dev).to_tf()
            q = tf.convert_to_tensor(q_in, dtype=tf.float32) #(4,)
            q = tf.expand_dims(q, 0) #(1,4)
            q_out.append(q)

        q_out = tf.concat(q_out, axis=0) #(128,4)

        return q_out

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def call(self, x):

        x = layers.Dense(self.n_qubits, activation='tanh')(x)

        # scale by PI/2
        x = layers.Lambda(lambda z: z*np.pi/2.)(x) 

        #print("After scaling:", x.shape)
        assert x.shape[1] == self.n_qubits

        q_out = self.run_quantum_circuit(x)

        q_out = layers.Dense(self.output_dim)(q_out)

        return q_out

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim)
        #print("compute_output_shape:", output_shape)

        return tf.TensorShape(output_shape)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_model_quantum(embed, n_categories, n_qubits=4, q_depth=6, embedding_dim=512):
    
    UniversalEmbedding = partial(USELayer,embed)

    text_in = keras.Input( shape=(1,), dtype=tf.string, name="text_in")

    x = layers.Lambda(UniversalEmbedding, output_shape=(embedding_dim,), name="USE_embedding")(text_in)

    q = dressed_quantum_circuit(
            output_dim=n_categories, 
            n_qubits=n_qubits, 
            q_depth=q_depth, 
            dynamic=True)(x)
    print("q shape ", q.shape)

    x_out = layers.Activation("softmax")(q)

    return keras.Model(inputs=text_in, outputs=x_out, name="AbstractClassifier")
