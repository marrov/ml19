
# coding: utf-8

# ## PDE reconstruction using (physics-informed) feedforward neural networks
# ## 1. Introduction
# 
# This example illustrates how to implement a neural network (NN) and modify the loss function to introduce some physical knowledge during the training of the network.
# The present case is illustrated on the Burgers' equation:
# \begin{equation}
# \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, x\in[-1,1]
# \end{equation}
# with Dirichlet boundary conditions:
# $$ u(-1,t) = u(1,t) = 0$$
# The initial condition is:
# $$u(x,0) = - \sin(\pi x)$$

# In this example, we will study how to implement a NN that can estimate the solution of the PDE presented above. To this aim, we will present a general implementation of a feedforward NN.

# ## 2. Code

# ### 2.1 Imports

# In[1]:


import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 7]
import scipy.io
from scipy.interpolate import griddata
# from pyDOE import lhs
import time
import h5py
# from Burgers_PINN import Burgers_PINN
np.random.seed(1234)
tf.set_random_seed(1234)


# #### Import the PINN class
# Have a look inside that notebook too

# In[2]:

from Burgers_PINN import *
# get_ipython().magic('run Burgers_PINN.ipynb')


# ### 2.1 Read the data from Burgers' equation
# The shock case: an initial sinusoide which evolves into a shock the initial condition is a -sin(pi*x)

# In[3]:


hf = h5py.File('./Data/burgers_shock.h5','r')
t = np.array(hf.get('/t')).T
x = np.array(hf.get('/x')).T
Exact = np.array(hf.get('/usol'))
hf.close()
nu = 0.01/np.pi
X, T = np.meshgrid(x,t)

X_all = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
Uexact_all = Exact.flatten()[:,None]              

# Domain bounds
lb = X_all.min(0)
ub = X_all.max(0)
plt.contourf(Exact)


# ### 2.3 Define the size of the NN
# For the feedforward NN, the number of layers and of neurons per layer is one of the most important parameters.
# It will describe (to some extent) the ability of the network to represent more complex function.

# In[4]:


layers = [2, 20, 20, 1]
model = Burgers_PINN(layers, lb, ub, nu)


# ### 2.4 Training of the NN on different datasets
# Here, we train the conventional NN on 2 types of data in the training dataset:
# 1. IBC case: Feedforward NN trained on just the knowledge of the initial and boundary conditions
# 2. data case: Feedforward NN trained on a random sample of data sample in the $[-1,1] \times [0,1]$ domain
# 
# For the physics informed neural network, we will also consider 2 types of data in the training dataset:
# 3. Physics + IBC case: Feedforward NN with the knowledge of the initial and boundary conditions and physical constraints on the residual
# 4. Physics + data case: Feedforward NN with a random sample of data and the physical constraints

# ### 2.4.1 Initial-BoundaryConditions case: Train the NN from points only on the IC and BC

# In[5]:


# Uniformly distributed points in x-domain [-1,1] for the initial condition
Nx_init = 50
x_init = np.linspace(lb[0],ub[0],Nx_init)
u_x_init = - np.sin(np.pi*x_init)

X_init = np.hstack((x_init[:,None],np.zeros(len(x_init))[:,None]) )

# Uniformly distributed points in t-domain [0,1] for the left BC
Nt_BC = 50
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
u_train = np.zeros(len(t_BC))
T_init = np.hstack((-np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_init,T_init))
u_trainIBC = np.vstack((u_x_init[:,None],np.zeros(len(t_BC))[:,None]) )

# Uniformly distributed points in t-domain [0,1] for the right BC
Nt_BC = 50
t_BC = np.linspace(lb[1],ub[1],Nt_BC)
u_train = np.zeros(len(t_BC))
T_init = np.hstack((np.ones(len(t_BC))[:,None],t_BC[:,None]) )
X_trainIBC = np.vstack((X_trainIBC,T_init))
u_trainIBC = np.vstack((u_trainIBC,np.zeros(len(t_BC))[:,None]) )


# In[6]:


# Training purely based on "data" (knowledge of the solution on the boundaries)
start_time = time.time()
model.train_data(X_trainIBC[:,0:1],X_trainIBC[:,1:],u_trainIBC)
elapsed_IBC = time.time() - start_time                
print('Training time: %.4f' % (elapsed_IBC))

# Prediction from the PINN on the [-1,1]x[0,1] domain
u_pred = model.predict_u(X_all[:,0:1],X_all[:,1:])

error_u_IBC = np.linalg.norm(Uexact_all-u_pred,2)/np.linalg.norm(Uexact_all,2)
print('Error u: %e' % (error_u_IBC))

U_pred_IBC = griddata(X_all, u_pred.flatten(), (X, T), method='cubic')

plt.figure()
plt.contourf(x[:,0],t[:,0],U_pred_IBC)
plt.plot(X_trainIBC[:,0],X_trainIBC[:,1],'rx', markersize=12)
plt.xlim([-1,1])
plt.ylim([0,1])


# ### 2.4.2 Data-case: Train the NN from randomly distributed points in the domain

# In[7]:


model.reset_weights()
# randomly picked points for the data-only training of the NN
Ntrain = 1000
idx = np.random.choice(X_all.shape[0], Ntrain, replace=False)
X_trainData = X_all[idx, :]
u_trainData = Uexact_all[idx,:]

start_time = time.time()
model.train_data(X_trainData[:,0:1],X_trainData[:,1:],u_trainData)
elapsed_Data = time.time() - start_time                
print('Training time: %.4f' % (elapsed_Data))

u_pred = model.predict_u(X_all[:,0:1],X_all[:,1:])
error_u_Data = np.linalg.norm(Uexact_all-u_pred,2)/np.linalg.norm(Uexact_all,2)
print('Error u: %e' % (error_u_Data))

U_pred_Data = griddata(X_all, u_pred.flatten(), (X, T), method='cubic')

plt.figure()
plt.contourf(x[:,0],t[:,0],U_pred_Data)
plt.plot(X_trainData[:,0],X_trainData[:,1],'rx', markersize=12)
plt.xlim([-1,1])
plt.ylim([0,1])


# ### 2.4.3 Phys-IBC case: Train the NN using the physical equation and the points on boundaries

# ### 2.4.4 Phys-Data case: Training of the PINN with random points in the domain

# ### 2.5 Comparison of the results and Discussion

# In[8]:


print('For IBC-case, error on u at datapoints is: %e' % (error_u_IBC))
print('For Data-case, error on u at all datapoints is: %e' % (error_u_Data))

plt.figure()
plt.subplot(1,4,1)
plt.contourf(x[:,0],t[:,0],U_pred_IBC)
plt.subplot(1,4,2)
plt.contourf(x[:,0],t[:,0],np.abs(U_pred_IBC-Exact))
plt.colorbar()

plt.subplot(1,4,3)
plt.contourf(x[:,0],t[:,0],U_pred_Data)
plt.subplot(1,4,4)
plt.contourf(x[:,0],t[:,0],np.abs(U_pred_Data-Exact))
plt.colorbar()
plt.show()


# ### 2.6 Save some results

# In[9]:


fln = 'Burgers_NN_Shock_results.h5'
hf = h5py.File(fln,'w')
hf.create_dataset('U_pred_IBC',data=U_pred_IBC)
hf.create_dataset('U_pred_Data',data=U_pred_Data)
hf.create_dataset('U_Exact',data=Exact)
hf.create_dataset('x',data=x)
hf.create_dataset('t',data=t)
hf.close()

