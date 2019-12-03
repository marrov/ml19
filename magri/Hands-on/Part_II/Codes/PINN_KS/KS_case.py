
# coding: utf-8

# # PDE reconstruction using (physics-informed) feedforward neural networks
# ## 1 Introduction

# This example illustrates how to implement a neural network (NN) and modify the loss function to introduce some physical knowledge during the training of the network.
# The present case is illustrated on the Kuramoto-Sivashinsky (KS) equation:
# \begin{equation}
# \frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial u}{\partial x^4} + u \frac{\partial u}{\partial x} = 0, \hspace{11pt}
# x\in[0,2\pi L]
# \end{equation}
# with periodic boundary conditions:
# The initial condition is:
# $$u(x,0) = - \sin(x/(2\pi L))$$
# and here $L=6$.
# The KS equation models the diffusive instabilities in a laminar flame front.

# In this example, we will study how to implement a NN that can estimate the solution of the PDE presented above. To this aim, we will present a general implementation of a feedforward NN and train it using 4 different set of points to show the effect of adding the physical constraints to the training of a feedforward NN.
#  - data case: Feedforward NN trained on a random sample of data sample in the $[0,2\pi L] \times [0,T]$ domain
#  - Physics + data case: Feedforward NN with a random sample of data and the physical constraints inside the domain and on the initial and boundary conditions.

# ## 2 Code
# ### 2.1 Imports

# In[1]:


import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import h5py

np.random.seed(1234)
tf.set_random_seed(1234)


# ### 2.2 Definition of the physics-informed neural network for Burgers' equation

# In[2]:

from KS_PINN import KS_PINN
# get_ipython().magic('run KS_PINN.ipynb')


# ### 2.3 Read the data from KS' equation

# In[3]:


hf = h5py.File('./Data/KS_data_L6_simple.h5')

t = np.array(hf.get('/t'))
x = np.array(hf.get('/x'))
Exact = np.array(hf.get('/u'))

X, T = np.meshgrid(x,t)

X_all = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
Uexact_all = Exact.flatten()[:,None]              

# Domain bounds
lb = X_all.min(0)
ub = X_all.max(0)


# In[4]:


plt.contourf(x,t,Exact)


# ### 2.4 Define the size of the NN

# In[5]:


layers = [2, 20, 20, 1]

# MODIFY THE KS_PINN class first
model = KS_PINN(layers, lb, ub)


# #### 2.4.1 Data-case: Train the NN from randomly distributed points in the domain

# In[6]:


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


# In[10]:


u_pred = model.predict_u(X_all[:,0:1],X_all[:,1:])

error_u_Data = np.linalg.norm(Uexact_all-u_pred,2)/np.linalg.norm(Uexact_all,2)
print('Error u: %e' % (error_u_Data))

U_pred_Data = griddata(X_all, u_pred.flatten(), (X, T), method='cubic')

plt.contourf(x,t,U_pred_Data)
plt.plot(X_trainData[:,0],X_trainData[:,1],'rx', markersize=12)


# #### 2.4.2 Phys-Data case: Train the NN using the physical equation, IC and BC and all the datapoints
# Modify to specify locations for the physical constraints (collocation, initial condition and boundary conditions)

# In[ ]:

"""
model.reset_weights()
# definition of the colocation points
# sampled for a latin-hypersampling method on the domain
N_coloc = 10000

# sample location for the IC
Nx_init = 50

# sample location for the periodic bc
Nt_BC = 50

start_time = time.time()

#model.train_phys(...) # provide the appropriate trace for the train_phys function

elapsed_PIBC = time.time() - start_time                
print('Training time: %.4f' % (elapsed_PIBC))


# In[ ]:


#u_pred = model.predict_u(X_all[:,0:1],X_all[:,1:])

#error_u_PINN = np.linalg.norm(Uexact_all-u_pred,2)/np.linalg.norm(Uexact_all,2)
#print('Error u: %e' % (error_u_PINN))

#U_pred_PINN = griddata(X_all, u_pred.flatten(), (X, T), method='cubic')

#plt.contourf(x,t,U_pred_PINN)

"""
# ## 3 Comparison of the results and Discussion

# In[ ]:


print('For Data-case, error on u at all datapoints is: %e' % (error_u_Data))
# print('For PIBC-case, error on u at all datapoints is: %e' % (error_u_PINN))



# In[ ]:


plt.figure()
plt.subplot(1,2,1)
plt.contourf(x,t,U_pred_Data)
plt.subplot(1,2,2)
plt.contourf(x,t,np.abs(U_pred_Data-Exact))
plt.colorbar()

"""
plt.figure()
plt.subplot(1,2,1)
plt.contourf(x,t,U_pred_PINN)
plt.subplot(1,2,2)
plt.contourf(x,t,np.abs(U_pred_PINN-Exact))
plt.colorbar()
"""

plt.figure()
plt.contourf(x,t,Exact)
plt.colorbar()
plt.show()

# In[ ]:


# Save some stuff
fln = 'KS_NN_results.h5'
hf = h5py.File(fln,'w')
hf.create_dataset('U_pred_data',data=U_pred_Data)
# hf.create_dataset('U_pred_PINN',data=U_pred_PINN)
hf.create_dataset('U_Exact',data=Exact)
hf.create_dataset('x',data=x)
hf.create_dataset('t',data=t)
hf.close()

