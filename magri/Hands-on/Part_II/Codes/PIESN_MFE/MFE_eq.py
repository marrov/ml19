
# coding: utf-8

# ## Module describing the physical equation of the system to study

# In[ ]:

import numpy as np
import tensorflow as tf


# In[ ]:


def MFE_RHS_tf(u, zeta,xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9):
    
    RHS = - tf.matmul(u,zeta)
    [u1,u2,u3,u4,u5,u6,u7,u8,u9] = tf.unstack(RHS,axis=1)
    
    u1 = u1 + zeta[0,0] - xi1[0]*u[:,5]*u[:,7] + xi1[1]*u[:,1]*u[:,2]
    
    u2 = u2 + xi2[0]*u[:,3]*u[:,5] - xi2[1]*u[:,4]*u[:,6] - xi2[2]*u[:,4]*u[:,7] - xi2[3]*u[:,0]*u[:,2] - xi2[4]*u[:,2]*u[:,8]
    u3 = u3 + xi3[0]*(u[:,3]*u[:,6] + u[:,4]*u[:,5] ) + xi3[1]*u[:,3]*u[:,7]
    u4 = u4 - xi4[0]*u[:,0]*u[:,4] - xi4[1]*u[:,1]*u[:,5] - xi4[2]*u[:,2]*u[:,6] - xi4[3]*u[:,2]*u[:,7] - xi4[4]*u[:,4]*u[:,8]
    u5 = u5 + xi5[0]*u[:,0]*u[:,3] + xi5[1]*u[:,1]*u[:,6] - xi5[2]*u[:,1]*u[:,7] + xi5[3]*u[:,3]*u[:,8] + xi5[4]*u[:,2]*u[:,5]
    u6 = u6 + xi6[0]*u[:,0]*u[:,6] + xi6[1]*u[:,0]*u[:,7] + xi6[2]*u[:,1]*u[:,3] - xi6[3]*u[:,2]*u[:,4] + xi6[4]*u[:,6]*u[:,8] + xi6[5]*u[:,7]*u[:,8]
    u7 = u7 - xi7[0]*(u[:,0]*u[:,5] + u[:,5]*u[:,8] ) + xi7[1]*u[:,1]*u[:,4] + xi7[2]*u[:,2]*u[:,3]
    u8 = u8 + xi8[0]*u[:,1]*u[:,4] + xi8[1]*u[:,2]*u[:,3]
    u9 = u9 + xi9[0]*u[:,1]*u[:,2] - xi9[1]*u[:,5]*u[:,7]
    
    #return [RHS[0], RHS[1], RHS[2], RHS[3], RHS[4], RHS[5], RHS[6], RHS[7], RHS[8] ]
    return tf.stack([u1,u2,u3,u4,u5,u6,u7,u8,u9],axis=1)

def MFE_step_tf(Uin, zeta,xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9, dt):
    
    #Y = np.zeros(Uin.shape)
    k1 = dt*MFE_RHS_tf(Uin, zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
    k2 = dt*MFE_RHS_tf(Uin+k1/3., zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
    k3 = dt*MFE_RHS_tf(Uin-k1/3.+k2, zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
    k4 = dt*MFE_RHS_tf(Uin+k1-k2+k3, zeta, xi1,xi2,xi3,xi4,xi5,xi6,xi7,xi8,xi9)
    
    Y = Uin + (k1 + 3.*k2 + 3.*k3 + k4) / 8.
    return Y


# In[ ]:


# return the reconstructed 2D velocity field in the specified y plane        
def get_MFE_2D(a, Lx, Lz, alpha, beta, gamma, y):
    Nx = 20
    Nz = 20
    Ny = 1
    xx = np.linspace(0,Lx,Nx)
    zz = np.linspace(0,Lz,Nz)
    #yy = [0.0]
    N8 = 2.*np.sqrt(2.) / np.sqrt((alpha**2+gamma**2)*(4*alpha**2+4*gamma**2+np.pi**2) )
    
    u = np.zeros((Nx,Nz,Ny,3))
    uu = np.zeros((3,9))
    #for iy in range(len(yy)):
    for ix in range(len(xx)):
        for iz in range(len(zz)):
            x = xx[ix]
            z = zz[iz]
            #y = yy[iy]
            uu[:,0] = [ np.sqrt(2.) * np.sin(np.pi*y/2.), 0, 0]
            uu[:,1] = [4/np.sqrt(3)* (np.cos(np.pi*y/2.))**2. * np.cos(gamma*z), 0, 0]
            uu[:,2] = (2./np.sqrt(4*gamma**2. + np.pi**2.) ) * np.array([0, 2.*gamma*np.cos(np.pi*y/2.)*np.cos(np.pi*z/2.), np.pi*np.sin(np.pi*y/2.)*np.sin(gamma*z) ])
            uu[:,3] = [0, 0, 4/np.sqrt(3) * np.cos(alpha*x) * (np.cos(np.pi*y/2.))**2.]
            uu[:,4] = [0, 0, 2.*np.sin(alpha*x)*np.sin(np.pi*y/2.)]
            uu[:,5]  = ( (4*np.sqrt(2.))/np.sqrt(3*(alpha**2.+gamma**2.)) ) * np.array([-gamma*np.cos(alpha*x)*(np.cos(np.pi*y/2.))**2.*np.sin(gamma*z), 0, alpha*np.sin(alpha*x)*(np.cos(np.pi*y/2.))**2.*np.cos(gamma*z)])
            uu[:,6]  = ( (2.*np.sqrt(2.))/np.sqrt(alpha**2.+gamma**2.) ) * np.array( [gamma*np.sin(alpha*x)*np.sin(np.pi*y/2.)*np.sin(gamma*z), 0, alpha*np.cos(alpha*x)*np.sin(np.pi*y/2.)*np.cos(gamma*z)])
            uu[:,7]  = N8* np.array([np.pi*alpha*np.sin(alpha*x)*np.sin(np.pi*y/2.)*np.sin(gamma*z),
                2.*(alpha**2.+gamma**2.)*np.cos(alpha*x)*np.cos(np.pi*y/2.)*np.sin(gamma*z),
                -np.pi*gamma*np.cos(alpha*x)*np.sin(np.pi*y/2.)*np.cos(gamma*z)])
            uu[:,8]  = [np.sqrt(2.)*np.sin(3*np.pi*y/2.), 0, 0]
            u[ix,iz,0,:] = np.matmul(uu,a)[:,0]
    return u

