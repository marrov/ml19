
# coding: utf-8

# # Echo State Network Class
# This notebook presents an implementation of the Echo State Network.
# 

# 
# Implementation inspired by
# 
# [1] "A practical Guide to Applying Echo State Networks", M. Lukosevicius in 
#     Neural Networks: tricks of the trade, Springer (2012).
# 
# [2] "Optimization and applications of echo state networks with leaky-integrator neurons", Jaeger et al.,
#     Neural networks, 20, 335-352 (2007).
# 
# [3] "Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach", Pathak et al.
#     Physical Review Letters, 120, 024102 (2018).
#     
# [4] "Hybrid forecasting of chaotic processes: Using machine learning in conjunction with a knowledge-based model", Pathak et al.
#     Chaos, 28, 041101 (2018).

# In[ ]:


# NAKD
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell_impl

import h5py

# to pass to py_func
def np_eigenvals(x):
    return np.linalg.eigvals(x).astype('complex128')

# Definition of the EchoState NN as a child of RNNCell of tensorflow
class EchoStateRNNCell(rnn_cell_impl.RNNCell):

    def __init__(self, num_units, num_inputs=1, decay=0.1, rho=0.6, 
                 sparseness=0.0,
                 sigma_in=1.0,
                 rng=None,
                 activation=None,
                 reuse = False,
                 win=None,
                 wecho=None):
        """
        Args:
            num_units: int, Number of units in the ESN cell.
            num_inputs: int, The number of inputs to the RNN cell.
            decay: float, Decay/leaking of the ODE of each unit.
            rho: float, Target spectral radius 1.
            sparseness: float [0,1], sparseness of the inner weight matrix.
            rng: np.random.RandomState, random number generator. (to be able to have always the same random matrix...)
            activation: Nonlinearity to use.
            reuse: if reusing existing matrix
            win: input weights matrix
            wecho: echo state matrix
        """

        # Basic RNNCell initialization (see tensorflow documentation)
        super(EchoStateRNNCell, self).__init__() # use the initializer of RNNCell (i.e. defines a bunch of standard variables)
        
        # fixed variables
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._num_inputs = num_inputs
        
        # variables that potentially can be changed/optimized (for future version)
        self.decay = decay
        self.rho = rho # rho has to be <1. to ensure the echo state property (see [2])
        self.sparseness = sparseness
        self.sigma_in = sigma_in
        
        # Random number generator initialization
        self.rng = rng
        if rng is None:
            self.rng = np.random.RandomState()
        
        # build initializers for tensorflow variables
        if (reuse == False):
            self.win = self.buildInputMatrix()
            self.wecho = self.buildEchoMatrix()
        else:
            self.win = win.astype('float64')
            self.wecho = wecho.astype('float64')
        
        # convert the weight to tf variable
        self.Win = tf.get_variable('Win',initializer = self.win, trainable = False)
        self.Wecho = tf.get_variable('Wecho', initializer = self.wecho, trainable = False)

        self.setEchoStateProperty()
 
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


    # function that has to be implemented in the tensorflow framework
    # return the output and new state of the NN for given inputs and current state of the NN
    def call(self, inputs, state):
        """ Echo-state RNN: 
            x = x + h*(f(W*inp + U*g(x)) - x).
        """
        
        new_state = state + self.decay*(
                self._activation(
                    tf.matmul(inputs, self.Win) +
                    tf.matmul(state, self.Wecho) 
                )
            - state)

        output = new_state

        return output, new_state   
    
    def setEchoStateProperty(self):
        """ optimize U to obtain alpha-improved echo-state property """
        # I know it's stupid for the time being but it is a placeholder for future treatment of the matrix
        # (potential meta-optimization and other)
        self.Wecho = self.normalizeEchoStateWeights(self.Wecho)
 
       
    # construct the Win matrix (dimension num_inputs x num_units)
    def buildInputMatrix(self):
        """            
            Returns:
            
            Matrix representing the 
            input weights to an ESN    
        """  

        # Input weight matrix initializer according to [3,4]
        # Each unit is connected randomly to a given input with a weight from a uniform distribution
        
        # without bias at the input
        #W = np.zeros((self._num_inputs,self._num_units))
        #for i in range(self._num_units):
            #W[self.rng.randint(0,self._num_inputs),i] = self.rng.uniform(-self.sigma_in,self.sigma_in)
        
        # Added bias in the input matrix
        W = np.zeros((self._num_inputs,self._num_units))
        for i in range(self._num_units):
            W[self.rng.randint(0,self._num_inputs),i] = self.rng.uniform(-self.sigma_in,self.sigma_in)
            
        # Dense input weigth [input] as in [1,2]
        # Input weigth matrix [input]
        #W = self.rng.uniform(-self.sigma_in, self.sigma_in, [self.num_inputs, self._num_units]).astype("float64")
        
        # Dense input weigth [bias, input] (as in [1,2])
        #W = self.rng.uniform(-self.sigma_in, self.sigma_in, [self.num_inputs+1, self._num_units]).astype("float64")
        
        return W.astype('float64')
    
    def getInputMatrix(self):
        return self.win
    
    def buildEchoMatrix(self):
        """            
            Returns:
            
            A 1-D tensor representing the 
            inner weights to an ESN (to be optimized)        
        """    

        # Inner weight tensor initializer
        # 1) Build random matrix from normal distribution between [0.,1.]
        #W = self.rng.randn(self._num_units, self._num_units).astype("float64") * \
                #(self.rng.rand(self._num_units, self._num_units) < (1. - self.sparseness) )
        
        # 2) Build random matrix from uniform distribution
        W = self.rng.uniform(-1.0,1.0, [self._num_units, self._num_units]).astype("float64") *                 (self.rng.rand(self._num_units, self._num_units) < (1. - self.sparseness) ) # trick to add zeros to have the sparseness required
        return W
    
    def normalizeEchoStateWeights(self, W):
        # Normalize to spectral radius rho

        eigvals = tf.py_func(np_eigenvals, [W], tf.complex128)
        W = W / tf.reduce_max(tf.abs(eigvals))*self.rho # sufficient conditions to ensure that the spectral radius is rho

        return W
    
    def getEchoMatrix(self):
        return self.wecho

    def save_ESN(self,fln,Wout=None,CurrState=None,lmb=None):
        
        hf = h5py.File(fln,'w')
        
        hf.create_dataset('Win',data=self.getInputMatrix() )
        hf.create_dataset('Wecho',data=self.getEchoMatrix() )
        hf.create_dataset('Wout',data=Wout)
        
        hf.create_dataset('rho',data=self.rho)
        hf.create_dataset('num_inputs',data=self._num_inputs)
        hf.create_dataset('num_units',data=self._num_units)
        hf.create_dataset('sigma_in',data=self.sigma_in)
        hf.create_dataset('sparseness',data=self.sparseness)
        hf.create_dataset('decay',data=self.decay)
        hf.create_dataset('lmb',data=lmb)
        hf.create_dataset('state',data=CurrState)
        
        hf.close()
                          

