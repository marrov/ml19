
# coding: utf-8

# ## Physics-Informed Neural Network class for the Burgers' equation
# 
# NB: Note that the class is here implemented over multiple cells by defining it recursively - this is _not_ a good coding practice... This is done for clarity in a jupyter notebook...

# In[1]:


import tensorflow as tf
import numpy as np

np.random.seed(1234)
tf.set_random_seed(1234)


# ## 1.1 Definition of the graph, loss, variables and optimizer of the PINN class

# In[2]:


class Burgers_PINN:
    # Define a Physics-Informed Neural Network which solves the Burgers' equation
    def __init__(self, layers, lb, ub, nu):
        # layers: list of the layers for the NN
        # lb: lower bounds of the domain
        # ub: upper bounds of the domain
        # nu: physical parameters (viscosity in the Burgers' equation)
        
        # defines the domain of prediction for the NN
        self.lb = lb
        self.ub = ub
        
        # defines the viscosity
        self.nu = nu
        
        # Metaparameters of the NN
        self.layers = layers
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # tf placeholders for the training data x,t are inputs, u are datapoints
        # the None dimension allows to have an unspecified dimension
        # (allowing to reuse the placeholder for different array size)
        self.x_train_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_train_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_train_tf = tf.placeholder(tf.float32, shape=[None, 1])
                
        self.u_pred = self.net_u(self.x_train_tf, self.t_train_tf) 
        
        # Loss function just based on the training set
        self.loss = tf.reduce_mean(tf.square(self.u_train_tf - self.u_pred))
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        
        # tf placeholders dedicated to physical constraints (collocation points)        
        self.x_coloc_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_coloc_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.f_pred = self.net_f(self.x_coloc_tf, self.t_coloc_tf)
        
        # Loss function based on the physical loss
        self.loss_phys = self.loss +                     tf.reduce_mean(tf.square(self.f_pred))
            
        self.optimizer_phys = tf.contrib.opt.ScipyOptimizerInterface(self.loss_phys,
                                                                    method = 'L-BFGS-B',
                                                                    options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        init = tf.global_variables_initializer()
        self.sess.run(init)


# ## 1.2 Initialization and reset methods of PINN

# In[3]:


class Burgers_PINN(Burgers_PINN):
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2./(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def reset_weights(self):
        reset = tf.global_variables_initializer()
        self.sess.run(reset)


# ## 1.3 Estimation and prediction methods of the PINN

# In[4]:


class Burgers_PINN(Burgers_PINN):
    # forward pass of the NN
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    # prediction of the NN at (x,t)
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    # estimate the prediction from the NN at points (x_pred,t_pred)
    def predict_u(self, x_pred, t_pred):
        u_hat = self.sess.run(self.u_pred, {self.x_train_tf: x_pred, self.t_train_tf: t_pred})  
        return u_hat


# The section below calls the optimizer to perform the training.

# In[5]:


class Burgers_PINN(Burgers_PINN):
    def callback(self, loss):
        print('Loss:', loss)
    
    # train the NN using the dataset defined by inputs=(x_train,t_train), outputs=(u_train)
    def train_data(self, x_train, t_train, u_train):
        tf_dict = {self.x_train_tf: x_train, self.t_train_tf: t_train,
                   self.u_train_tf: u_train}
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)
        


# ## 1.4 Additional physics-related estimation

# In[6]:


class Burgers_PINN(Burgers_PINN):
    # estimation of physical residual at (x,t)
    def net_f(self, x,t):
        # we make here use of the capability of tf to automate the gradient computation
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - self.nu*u_xx
        return f
    
    # estimate the physical residual
    def predict_f(self,x_pred, t_pred):
        f_hat = self.sess.run(self.f_pred, {self.x_coloc_tf: x_pred, self.t_coloc_tf: t_pred})
        return f_hat
        
    # train the NN using the dataset defined by inputs=(x_train,t_train), outputs=(u_train)
    # and minimizing the physical constraints at points (x_coloc,t_coloc)
    def train_phys(self, x_train, t_train, u_train, x_coloc, t_coloc):
        tf_dict = {self.x_train_tf: x_train, self.t_train_tf: t_train,
                   self.u_train_tf: u_train,
                  self.x_coloc_tf: x_coloc, self.t_coloc_tf: t_coloc}
                                                                                                                          
        self.optimizer_phys.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss_phys], 
                                loss_callback = self.callback)
    

