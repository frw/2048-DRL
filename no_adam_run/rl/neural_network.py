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
        self.momentum_coeff = 0.9

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

        
        # old version of stochastic gradient descent
        gparams = [T.grad(cost, wrt=param) for param in architecture.params]
        updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

        #stochastic gradient descent with adaptive learning using lasagne--take your pick
        #updates_sgd = sgd(cost, architecture.params, learning_rate=self.learning_rate)
        #updates = apply_momentum(updates_sgd, architecture.params, momentum=self.momentum_coeff)
        #updates = adadelta(cost, architecture.params, learning_rate=self.learning_rate, rho=0.95, epsilon=1e-06)
        #updates = adam(cost, architecture.params, learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)

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
        state_action_rep = self.generate_network_inputs(current_state, current_action)
        self.train_model(state_action_rep, target_value)

        #extra test
        #cost, result = self.train_model(state_action_rep, target_value)
        #return cost, result

    def use_model (self, current_state, current_action):

        state_action_rep = self.generate_network_inputs(current_state, current_action)
        return self.run_model(state_action_rep)

    def generate_network_inputs (self, raw_state, raw_action):
        '''
        Transforms state/action pair into input features for the network.
        '''
        state_action_rep = np.zeros(20)
        state_action_rep[:16] = np.asarray(raw_state)
        state_action_rep[16 + raw_action] = 1.0
        return state_action_rep

    def get_all_weights (self):
        '''
        Returns all network weights. Format: list of 2-D (inter-node weights) 
        numpy arrays and 1-D (bias term) numpy arrays.
        '''
        return self.grab_weights()





#Some haphazard extra code to test the neural network.
#my_nn = QNetwork()
'''for i in range(5):
    hello, yo = my_nn.update_model((1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1), 2, 3)
    #test_array = np.ones(20)
    #hello = my_nn.use_model(test_array)
    #print hello
    #print yo
    hello = my_nn.use_model((1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1), 2)
    #print hello
    #print my_nn.get_all_weights()'''
'''test_array = np.ones(20)
tester = np.ones(20) * 2.0
for i in range(5000):
    hello,hello2 = my_nn.update_model(test_array,20.0)
    print hello
    print hello2
    h1, h2 = my_nn.update_model(tester,10.0)
    print h1
    print h2'''
'''
x_data3 = np.array([1,2,3,4,5])
x_data2 = np.random.permutation(x_data3)
x_data = np.tile(x_data2,(20,1))
y_data = x_data2 * 3.0 #+ (np.random.normal(5) * 0.01)
print y_data
for k in range(500000):
    for i in range(y_data.shape[0]):
        next_step, n2 = my_nn.train_model(x_data[:,i],y_data[i])
        result = my_nn.run_model(x_data[:,i])
        if k % 1000 == 0:
            print "cost:"
            print next_step
            print "prediction:"
            print n2
            print "actual:"
            print y_data[i]
            print "try by forward:"
            print result
''''''
x_data3 = np.array([1,2,3,4,5])
x_data2 = np.random.permutation(x_data3)
x_data = np.tile(x_data2,(16,1))
y_data = x_data2 * 3.0 #+ (np.random.normal(5) * 0.01)
print y_data
for k in range(500000):
    for i in range(y_data.shape[0]):
        next_step, n2 = my_nn.update_model(x_data[:,i], 2, y_data[i])
        result = my_nn.use_model(x_data[:,i], 2)
        if k % 1000 == 0:
            print "cost:"
            print next_step
            print "prediction:"
            print n2
            print "actual:"
            print y_data[i]
            print "try by forward:"
            print result'''
