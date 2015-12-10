'''
The Deep Q-Learner's neural network. Coded using Theano.
See the final paper for the specifications of this baseline neural network using Adam.
Additional changes were made to this baseline when testing the different methods.
'''

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.updates import sgd, apply_momentum, adadelta, adam
from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        # initialize weights into this layer
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    size=(n_in, n_out),
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        # initialize bias term weights into this layer
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]


class OutputLayer(HiddenLayer):
    def error_function(self, y):
        return T.mean((self.output - y)**2.0)

    def get_result(self):
        return self.output


class Architecture(object):
    '''
    Defines architecture of network. Baseline uses fully connected network
    with one hidden layer.
    '''
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.outputLayer = OutputLayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=None
        )

        # L1 norm regularization
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm regularization
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        self.error_function = (
            self.outputLayer.error_function
        )

        self.params = self.hiddenLayer.params + self.outputLayer.params

        self.input = input

    def get_result(self):
        return self.outputLayer.get_result()


class QNetwork(object):
    '''
    Setup of neural network functionality, including parameters,
    the error function, forward passes, and backpropogation.
    '''
    def __init__ (self):
        self.learning_rate = 0.001
        self.L1_reg = 0.0000
        self.L2_reg = 0.0001
        self.n_hidden = 50
        self.num_inputs = 20
        self.num_outputs = 1

        # allocate symbolic variables for the data
        x = T.ivector('x')  
        y = T.iscalar('y') 

        rng = np.random.RandomState(None)

        # construct the neural network's Architecture
        architecture = Architecture(
            rng=rng,
            input=[x],
            n_in=self.num_inputs,
            n_hidden=self.n_hidden,
            n_out=self.num_outputs
        )

        cost = (
            architecture.error_function(y)
            + self.L1_reg * architecture.L1
            + self.L2_reg * architecture.L2_sqr
        )
        
        #stochastic gradient descent with adaptive learning with Lasagne--using Adam
        updates = adam(cost, architecture.params, learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)

        # backpropogation that also contains a forward pass
        self.train_model = theano.function(
            inputs=[x, y],
            outputs=[cost, architecture.get_result()],
            updates=updates,
            allow_input_downcast=True
        )

        # forward pass
        self.run_model = theano.function(
            inputs=[x],
            outputs=architecture.get_result(),
            allow_input_downcast=True
        )

        self.grab_weights = theano.function(
            inputs=[],
            outputs=architecture.params,
            allow_input_downcast=True
        )

    def update_model (self, current_state, current_action, target_value):
        '''
        Calls function to do backpropogation, given the board state, action, and target
        Q-value information from the reinforcement learner.
        '''
        state_action_rep = self.generate_network_inputs(current_state, current_action)
        self.train_model(state_action_rep, target_value)

    def use_model (self, current_state, current_action):
        '''
        Calls function to do forward pass. Yields a Q-value approximation from the network
        for a given board state and action.
        '''
        state_action_rep = self.generate_network_inputs(current_state, current_action)
        return self.run_model(state_action_rep)

    def generate_network_inputs (self, raw_state, raw_action):
        '''
        Transforms 2048 state/action pair into input features for the network.
        '''
        state_action_rep = np.zeros(20)
        state_action_rep[:16] = np.asarray(raw_state)
        state_action_rep[16 + raw_action] = 1.0
        return state_action_rep

    def get_all_weights (self):
        '''
        Returns all network weights. Format: list of 2-D (inter-node weights) 
        numpy arrays and 1-D (bias term) numpy arrays. See plot.py function
        for more in-depth description.
        '''
        return self.grab_weights()