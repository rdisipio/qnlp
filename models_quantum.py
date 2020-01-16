import numpy as np

import pennylane as qml

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from functools import partial
from models import USELayer

'''
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
        super(dressed_quantum_circuit, self).__init__(dynamic=True, **kwargs)

        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build(self, input_shape):
        print("Input shape:", input_shape)
        
        self.W1 = self.add_weight(name='W1', 
                                    shape=(input_shape[-1], self.n_qubits),
                                    initializer='uniform',
                                    trainable=True)
        
        self.b1 = self.add_weight(name='b1',
                                    shape=(self.n_qubits,),
                                    initializer='uniform',
                                    trainable=True)
        
        self.q_weights = self.add_weight(name='q_weights',
                                    shape=(self.q_depth, self.n_qubits),
                                    initializer='uniform',
                                    trainable=True)
        
        self.W2 = self.add_weight(name='W2',
                                    shape=(self.n_qubits, self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
        
        self.b2 = self.add_weight(name='b2',
                                    shape=(self.output_dim,),
                                    initializer='uniform',
                                    trainable=True)
        
        super(dressed_quantum_circuit, self).build(input_shape)  # Be sure to call this at the end

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

    def variational_quantum_circuit(self, q_in):

        q_weights = self.q_weights.numpy()

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

    def call(self, inputs):

        x = tf.matmul(inputs, self.W1) + self.b1
        x = tf.math.tanh(x)

        #assert x.shape[-1] == self.n_qubits

        # scale by PI/2
        x = tf.math.scalar_mul(np.pi/2., x) 

        q = self.run_quantum_circuit(x)

        z = tf.matmul(q, self.W2) + self.b2

        #assert z.shape[-1] == self.output_dim

        return z

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[-1], self.output_dim)

        return tf.TensorShape(output_shape)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_model_quantum_V1(embed, n_categories, n_qubits=4, q_depth=6):
    
    UniversalEmbedding = partial(USELayer, embed)

    text_in = keras.Input( shape=(1,), dtype=tf.string, name="text_in")

    x = layers.Lambda(UniversalEmbedding, name="USE_embedding")(text_in)
    print("USE:", x)
    x = dressed_quantum_circuit(
            output_dim=n_categories, 
            n_qubits=n_qubits, 
            q_depth=q_depth)(x)
    
    assert x.shape[-1] == n_categories

    #x = layers.Dense(2)(x)
    x_out = layers.Activation("softmax")(x)

    return keras.Model(inputs=text_in, outputs=x_out, name="QuantumPreprintClassifier")

'''

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


def make_model_quantum(n_categories, n_qubits=4, n_layers=2, embedding_dim=512):
    
    '''
    UniversalEmbedding = partial(USELayer, embed)
    text_in = keras.Input( shape=(1,), dtype=tf.string, name="text_in")
    x = layers.Lambda(UniversalEmbedding, name="USE_embedding", dtype=tf.float64)(text_in)
    '''

    text_in = keras.Input( shape=(embedding_dim,), dtype=tf.float64, name='text_in')

    x = layers.Dense(n_qubits, activation='tanh', dtype=tf.float64)(text_in)

    x = VariationalQuantumCircuit(
            n_categories=n_categories, 
            n_qubits=n_qubits, 
            n_layers=n_layers)(x)
    
    assert x.shape[-1] == n_qubits

    x_out = layers.Dense(n_categories, activation='softmax', dtype=tf.float64)(x)
    #x = layers.Dense(2)(x)
    #x_out = layers.Activation("softmax")(x)

    return keras.Model(inputs=text_in, outputs=x_out, name="QuantumPreprintClassifier")