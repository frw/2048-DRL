import cPickle
import gzip
import os
import sys

import numpy as np

import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

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

    def __init__ (self):
        self.learning_rate = 0.001
        self.L1_reg = 0.0000
        self.L2_reg = 0.0001
        self.batch_size = 20
        self.n_hidden = 50
        self.num_inputs = 20
        self.num_outputs = 1

        # allocate symbolic variables for the data
        x = T.ivector('x')  
        y = T.iscalar('y')  

        rng = np.random.RandomState(None)

        # construct the neural network's Architecture
        classifier = Architecture(
            rng=rng,
            input=x,
            n_in=self.num_inputs,
            n_hidden=self.n_hidden,
            n_out=self.num_outputs
        )

        cost = (
            classifier.error_function(y)
            + self.L1_reg * classifier.L1
            + self.L2_reg * classifier.L2_sqr
        )

        # compute the gradient of cost with respect to all weights
        gparams = [T.grad(cost, wrt=param) for param in classifier.params]

        # apply gradient descent
        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
        ]

        # backpropogation that also contains a forward pass
        self.train_model = theano.function(
            inputs=[x, y],
            outputs=[cost, classifier.get_result()],
            updates=updates,
            allow_input_downcast=True
        )

        # forward pass
        self.run_model = theano.function(
            inputs=[x],
            outputs=classifier.get_result(),
            allow_input_downcast=True
        )

    def update_model (self, state_action_rep, target_value):
        return self.train_model(state_action_rep, target_value)

    def use_model (self, state_action_rep):
        return self.run_model(state_action_rep)


my_nn = QNetwork()





#Some haphazard extra code to test the neural network.
'''
test_array = np.ones(20)
tester = np.ones(20) * 2.0
for i in range(500):
    hello,hello2 = my_nn.update_model(test_array,20.0)
    print hello
    print hello2
    h1, h2 = my_nn.update_model(tester,10.0)
    print h1
    print h2
x_data3 = np.array([1,2,3,4,5])
x_data2 = np.random.permutation(x_data3)
x_data = np.tile(x_data2,(20,1))
y_data = x_data2 * 3.0 + (np.random.normal() * 0.01)
print y_data
for k in range(50000):
    for i in range(y_data.shape[0]):
        next_step, n2 = my_nn.update_model(x_data[:,i],y_data[i])
        result = my_nn.use_model(x_data[:,i])
        if k % 1000 == 0:
            print next_step
            print n2
            print result
            print y_data[i]
'''