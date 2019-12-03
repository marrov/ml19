
# coding: utf-8

# ## Modelling and prediction of chaotic systems using Echo State Network 

# ## 1. Introduction

# In this notebook, we will study how to use the ESN to learn the dynamics of a model for shear flow, called the MFE model [1]. This latter model is derived from the Navier-Stokes equation based on a modal decomposition. The objective is thus to track the evolution of the 9 modes of the MFE equation.
# 
# [1] Moehlis, J., Faisst, H., & Eckhardt, B. (2004). A low-dimensional model for turbulent shear flows. New Journal of Physics, 6. https://doi.org/10.1088/1367-2630/6/1/056

# ## 2. Code

# ### 2.1 Library import

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import math_ops
import h5py
import time
from util import *
from mpl_toolkits import mplot3d
#from ESN import EchoStateRNNCell
#from Lorenz_eq import *


# Import the physical equation for the MFE system

# In[2]:

from MFE_eq import *
#get_ipython().magic('run MFE_eq.ipynb')


# ### 2.2 Import the ESN class
# See ESN.ipynb for the implementation of the ESN.
# The current implementation uses the Recurrent Neural Network class existing in tensorflow for convenience.
# Details of the exact architecture of the ESN are given in the course notes.

# In[3]:

from ESN import EchoStateRNNCell
#get_ipython().magic('run ESN.ipynb # import the ESN class')


# Set random seeds for reproducibility of runs

# In[4]:


# Set random seeds for reproducibility of runs
# random numbers
random_seed = 1
rng = np.random.RandomState(random_seed)


# ### 2.3 Read and treatment of the data
# The data is a relatively long run of the Lorenz system which was generated using an explicit Euler scheme with $\Delta t = 0.01$.

# In[5]:


hf = h5py.File('./Data/MFE_Re600_T30000.h5','r') # this data has 20000 timestep, DT=0.25, T=5000 and Nx=48
case = 'MFE'
u_exact = np.array(hf.get('u'))
t = np.array(hf.get('t'))
dt = np.array(hf.get('dt'))

# physical parameters for the MFE model
Lx = np.array(hf.get('Lx'))
Lz = np.array(hf.get('Lz'))
alpha = np.array(hf.get('alpha'))
beta = np.array(hf.get('beta'))
gamma_mod = np.array(hf.get('gamma'))
zeta = np.array(hf.get('zeta'))
xiall = []
for i in range(1,10):
    var = 'xi' + str(i)
    xiall.append( np.array(hf.get(var)) )

[xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9] = xiall
del xiall

hf.close()

num_inputs = u_exact.shape[1]
Nx = u_exact.shape[1]


# In[6]:


u_2D = get_MFE_2D(u_exact[0:1,:].T, Lx, Lz, alpha, beta, gamma_mod, 0.0)
plt.figure()
plt.contourf(u_2D[:,:,0,1])


# One can see from this trajectory in phase space that the Lorenz system "orbits" around 2 attractors and randomly jumps from one orbit to the other.
#  
# For what follows, only a small subset of that dataset will be used for the training of the ESN.

# ### Split of the data into training part
# Here, we put ourselves in the small data paradigm.
# As a result, we will only keep 1000 timesteps which roughly corresponds to 10 Lyapunov time for the Lorenz system.
# 
# We will use the following terminology for the split of the data:
#  - Training: data with known inputs/outputs used to "teacher-forced" the ESN; the outputs of the ESN are collected and used to compute $W_{out}$.
#  - Validation: data with known inputs/outputs; the inputs is fed to the ESN and the outputs of the ESN are compared to the real outputs for validation of the computation of $W_{out}$ (this is still a teacher-forced situation).
#  - prediction: The ESN is loopbacked to itself to provide a natural response; this natural response is compared to the exact evolution of the Lorenz system for the same initial conditions.

# In[7]:


## Parameters for the split of the training/validation ----------------------------
batches = 1 # number of batches for training
stime = 5000 # total sample size for training + validation

begin = 100 # begin of training (wash out at the start to remove the transient of the reservoir)
end = 5000 # has to be smaller than stime
horizon = 500 # duration of the prediction

cut1 = 46000 # to get rid of the initial transient in the simulation
cut2 = 46000+stime+horizon+1 # 
u_exact = u_exact[np.arange(cut1,cut2),:] # get rid of some initial transients

### Treatment of inputs -----------------------------------------------------------
rnn_inputs = np.zeros((batches, stime, num_inputs), dtype="float64")
wave = u_exact[:stime+1,:].astype("float64")

rnn_inputs = wave[:-1,:].reshape(1,stime, num_inputs)

### Treatment of output ------------------------------------------------------------
num_outputs = Nx ## 
rnn_target = wave[1:,:]
rnn_target = rnn_target.reshape(stime, num_outputs).astype("float64")

del wave


# ### 2.4 Definition of the metaparameters for the ESN
# The metaparameters of the ESN are defined in the course notes.

# In[8]:


### ESN metaparameters
num_units = 200
decay = 1.0 # for leakage (not used in PRL/chaos paper)
rho_spectral = 0.6 # spectral radius of Wecho
sigma_in = 1.0 # scaling of input weight
sparseness = 1. - 3. / (num_units - 1.) # sparseness of Wecho
# it's defined as 1 - degree / (num_units - 1) where degree is the average number of connections of a unit to other units in reservoir
lmb = 1e-6 
activation = lambda x: math_ops.tanh(x) # the activation function of the ESN.ipynb


# ### 2.5 Definition of the tensorflow graph

# In[9]:


tf.reset_default_graph()
graph = tf.Graph()

rng = np.random.RandomState(random_seed)
# Initialize the ESN cell
cell = EchoStateRNNCell(num_units=num_units, 
                        num_inputs=num_inputs,
                        activation=activation, 
                        decay=decay, # decay (leakage) rate
                        rho=rho_spectral, # spectral radius of echo matrix
                        sigma_in=sigma_in, # scaling of input matrix
                        sparseness = sparseness, # sparsity of the echo matrix
                        rng=rng)


# ### Training/Teacher-forced part of the graph
# In this part of the graph, the inputs of the ESN are provided - hence it is called a "teacher-forced" training.
# This test ultimately just allows to assess the "one-step prediction" capability of the ESN, i.e., whether its prediction $\widehat{y}(t+\Delta t)$ given $u(t)$ is close to $y(t+\Delta t)$.

# In[10]:


inputs = tf.placeholder(tf.float64, [batches, None, num_inputs])
init_state = tf.placeholder(tf.float64, [1, num_units])

Ytarget_tf = tf.placeholder(tf.float64,[None, num_outputs])

# Build the graph (for training on a time sequence - teacher forcing)
outputs,_ = tf.nn.dynamic_rnn(cell=cell,inputs=inputs,initial_state=init_state,dtype=tf.float64)
#outputs = tf.reshape(outputs, [stime, num_units])   
outputs = outputs[0,:,:]

Wout_tf = tf.Variable(np.random.rand(num_units,num_outputs),dtype=tf.float64,trainable=True)
Ytrain_tf = tf.matmul(outputs,Wout_tf)


# ### Natural prediction part of the graph
# In this part of the graph, the output of the ESN is looped back as its input to obtain its "autonomous response",
# i.e. $\widehat{y}(t+\Delta t)$ given $\widehat{y}(t)$ as input.
# This autonomous response is compared to the exact evolution in the data and
# this comparison allows to examine whether the trained ESN is a good approximation of the Lorenz system.

# In[11]:


# Build the part of the graph for future prediction (natural response of the ESN)
# This is made so as to have an "entry" points via the inputF,stateF variables
inputF = tf.reshape(Ytrain_tf[-1,:],[1, num_inputs])
stateF = tf.placeholder(tf.float64, [1, num_units])
stateL,_ = cell(inputs=inputF, state=stateF)
#Wout_tf = tf.placeholder(tf.float64, [num_units, num_outputs])
Ypred_tf = tf.matmul(stateL,Wout_tf)

# Build the graph extension for prediction
state_pred = stateL
Ypred_all_tf = []
Ypred_all_tf.append(Ypred_tf)
for it in range(1,horizon):
    state_pred,_ = cell(inputs=Ypred_tf,state=state_pred)
    Ypred_tf = tf.matmul(state_pred,Wout_tf)
    Ypred_all_tf.append(Ypred_tf)
outputs_pred = tf.reshape(Ypred_all_tf, [horizon, num_outputs])


# ### 2.6 Data-only training of the ESN
# This section has 3 major parts:
# 1. Get the states associated to the input time series
# 2. Compute $W_{out}$ via ridge regression
# 3. Compute the output of the ESN using the $W_{out}$ compute in 2.

# In[12]:


config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# training of the ESN
rnn_init_state = np.zeros([batches, num_units], dtype="float64")

stored_outputs  = sess.run(outputs, feed_dict={inputs:rnn_inputs,
                                        init_state:rnn_init_state}) # get the reservoir states for the input

XX_train = stored_outputs[begin:end,:].astype('float64') # Washout
st =  np.reshape(stored_outputs[-1,:],[1, num_units]).astype('float64') # save the last state for future prediction

# compute via Ridge regression, save and load the resulting Wout
Wout = np.matmul( 
    np.linalg.inv(np.matmul(np.transpose(XX_train), XX_train) +
        lmb*np.eye(num_units)),
    np.matmul(np.transpose(XX_train), rnn_target[begin:end,:]) )
Wout0 = Wout
Wout_tf.load(Wout,sess)


# In[13]:


## get the estimation from the ESN on training data
Yout = sess.run(Ytrain_tf,feed_dict={outputs:stored_outputs})


# In[14]:

plt.figure()
plt.plot(t[begin:end],rnn_target[begin:end,:])
plt.plot(t[begin:end],Yout[begin:end,:],'--')


# #### Teacher/input-forced ESN

# In[15]:


new_input = u_exact[stime+1:stime+100+1,:]
new_input = new_input.reshape(1,new_input.shape[0],new_input.shape[1])
Ytest = sess.run(Ytrain_tf,feed_dict={inputs:new_input,
                                        init_state:rnn_init_state})

plt.figure()
plt.plot(u_exact[stime+2:stime+horizon+2,:])
plt.plot(Ytest,'--')

# Similarly, when the ESN is in an input-forced situation, the ESN provides excellent agreement with the actual evolution of the Lorenz system.


# #### Natural response of the ESN

# In[16]:


inn = Yout[-1,:]
# future prediction
inn = inn.reshape((1,num_outputs)).astype('float64')

Ypred_ESN =  sess.run(outputs_pred,feed_dict={stateF:st, inputF:inn})
Epred_ESN = np.sum((Ypred_ESN-u_exact[stime+1:stime+horizon+1,:num_outputs])**2.,1)  / np.mean((u_exact[stime+1:stime+horizon+1,:num_outputs]**2.))

plt.figure()
plt.plot(t[end:end+horizon],u_exact[stime+1:stime+horizon+1,:])
plt.plot(t[end:end+horizon],Ypred_ESN,'--')
plt.figure()
plt.plot(t[end:end+horizon],Epred_ESN)

# However, when looking at the natural response of the ESN, we can see that discrepancies between the ESN and the Lorenz system appears quite fast.


# ### 2.7 Physics informed part

# In[17]:


# def Lorenz_step_tf(Uin,sigma,rho,beta,dt):


# We define here a new graph where the natural response of the ESN is computed and the values collected. The physical residual of this series is then computed and a loss function is defined based on the error on the training data and this physical residual. This loss function is then used in an optimizer to compute a new  Wout.

# In[18]:


# Build the graph extension for validation/optimization
Yval_tf = tf.matmul(stateL,Wout_tf)
state_val = stateL
valid_hor = 1000

## code the graph extension hereunder

# LOSS_TF = ...


# Define the optimizer which minimize LOSS_TF

# In[19]:


# Define and initialize the optimizer
#optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(LOSS_TF)
#initialize_uninitialized(sess)


# In[ ]:


## MODIFY THIS SECTION ACCORDING TO THE WAY YOU DEFINE THE LOSS_TF
## AND PROVIDE THE APPROPRIATE feed_dict to the sess.run

#opt_res = sess.run(LOSS_TF,feed_dict ={...})
#print(opt_res)

#for iopt in range(100):
#    sess.run(optimizer, feed_dict ={...})
#    opt_res = sess.run(LOSS_TF,feed_dict ={...})
#    if (iopt%50)==0:
#        print(opt_res)

#plt.rcParams["figure.figsize"] = (15,7)
plt.figure()
plt.plot(t[end:end+horizon],u_exact[stime+1:stime+horizon+1,:])
plt.plot(t[end:end+horizon],Ypred_ESN,'--')
plt.figure()
plt.plot(t[end:end+horizon],Epred_ESN)

plt.show()


# In[ ]:

"""
Ypred_PIESN =  sess.run(outputs_pred,feed_dict={stateF:st, inputF:inn})
Epred_PIESN = np.sum((Ypred_PIESN-u_exact[stime+1:stime+horizon+1,:num_outputs])**2.,1)  / np.mean((u_exact[stime+1:stime+horizon+1,:num_outputs]**2.))
plt.rcParams["figure.figsize"] = (15,7)
plt.figure()
plt.subplot(1,2,1)
plt.plot(t[end:end+horizon],u_exact[stime+1:stime+horizon+1,:])
plt.plot(t[end:end+horizon],Ypred_PIESN,'--')
plt.subplot(1,2,2)
plt.plot(t[end:end+horizon],u_exact[stime+1:stime+horizon+1,:])
plt.plot(t[end:end+horizon],Ypred_ESN,'--')
plt.figure()
plt.plot(t[end:end+horizon],Epred_PIESN,'--')
plt.plot(t[end:end+horizon],Epred_ESN)
"""

# ### Save results and trained ESN
# The saving of the ESN consists in saving the $W_{in}$, $W$ and $W_{out}$ matrices and of the hyperparameters of the ESN and the random seed.
# This ESN can be re-used for other predictions.

# In[ ]:


fln = 'MFE_case.h5'
hf = h5py.File(fln,'w')
hf.create_dataset('Epred_ESN',data=Epred_ESN)
hf.create_dataset('Ypred_ESN',data=Ypred_ESN)
hf.create_dataset('Yexact',data=u_exact[stime+1:stime+horizon+1,:])
hf.close()

fln_mod = 'MFE_model.h5'
cell.save_ESN(fln_mod,Wout,st,lmb)
hf = h5py.File(fln_mod,'a')
hf.create_dataset('Wout0',data=Wout0)
hf.close()

