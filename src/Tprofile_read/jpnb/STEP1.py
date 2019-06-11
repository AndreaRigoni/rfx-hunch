
# coding: utf-8

# # AEFIT
# 
# This would be the first attempt to run the unsupervised learning VAE network to learn how to characterize a 1D profile with atted noise and missing input.
# 
# More than a simple fit, this method should learn 

# In[1]:


import numpy as np
import tensorflow as tf

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import ipysh
import Hunch_utils  as Htls
import Hunch_lsplot as Hplt

import Dummy_g1data as dummy
import models.AEFIT as aefit

ipysh.Bootstrap_support.debug()


# ## Data and Model
# The model and data generator are set:
# Dummy data generator generates from a set of 5 kind of curves with a dataset cardinality of 60K samples.
# 
# All the shapes are generated from a dictionary array that defines mean sigma and gain of sum of gaussians.
# This table is printed from the variable ds.kinds
# 
# >NOTE: 
# > The actual model is generating random so it is not redoing the very same samples on each epoch.
# > To exactly constraint the maximum size of the dataset the tf buffer can be used
# 
# the model uses bby default an input of 40 samples that are the (x,y) tuple values of 20 points from the generated shapes.
# 

# In[17]:


ds = dummy.Dummy_g1data(counts=60000)
m = aefit.AEFIT(latent_dim=2)

ds.kinds


# In[3]:


# aefit.test_dummy(m, data=ds, epoch=5, batch=200, loss_factor=1e-3)


# In[4]:


# m.save('kcp/aefit1')


# In[18]:


m.load('kcp/prova1')





def simulate_missing_data(lpt=[0.5,-1.6], noise_var=0.05, arr = [3,2,1,5,8,7,6,9,12,11,14,13,18]):
    xy = m.decode(tf.convert_to_tensor([pt]),apply_sigmoid=True)
    x,y = tf.split(xy[0], 2)
    x,y = (x.numpy(), y.numpy())

    fig = plt.figure('gen_missing_curve',figsize=(18, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)    
    
    ax1.set_xlim(-2.,2.)
    ax1.set_ylim(-2.,2.)
    
    ax1.scatter(pt[0],pt[1],s=80)
    ax2.scatter(x,y,s=40)

    # apply noise
    x += np.random.normal(0,noise_var,len(x))
    y += np.random.normal(0,noise_var,len(y))

    # apply missing data simulation
    for i,v in enumerate(arr,0):
        x[arr[i]]=x[arr[i]+1]
        y[arr[i]]=y[arr[i]+1]
    
    ax2.scatter(x,y,s=80)

    me,va = m.encode(tf.reshape(tf.concat([x,y],0), shape=[1,-1]))
    print("Guessed Latent point = ",me.numpy())
    gpt = me[0].numpy()
    ax1.scatter(gpt[0],gpt[1])
    
    XY = m.decode(me,apply_sigmoid=True)
    X,Y = tf.split(XY[0], 2)
    X,Y = (X.numpy(), Y.numpy())
    # plt.figure('reconstructed')
    ax2.scatter(X,Y,s=40)
    # plt.plot(X,Y)




    
# generate from point: 0.6, -0.7
pt = [0.6,-0.7]
noise_var = 0.05
arr = [3,2,1,5,8,7,6,9,12,11,14,13,18]
simulate_missing_data(pt,noise_var,arr)


# generate from point: 0.5, -1.6
pt = [0.5,-1.6]
noise_var = 0.05
arr = [3,2,1,5,8,7,6,9,12,11,14,13,18]
simulate_missing_data(pt,noise_var,arr)


