import tensorflow as tf
import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt
from math import sqrt

# init
X = []
Y = []

# approximate a function
'''
X = np.random.rand(20,2)
Y = 2*X[:,[0]]+X[:,[1]]
'''

# .. or approximate data
if os.path.isfile("./trial/input.tmp") and os.path.isfile("./trial/output.tmp"):
    print("Reading data...")
    # Import input and output data
    X = np.loadtxt("./trial/input.tmp", ndmin=2)
    Y = np.loadtxt("./trial/output.tmp", ndmin=2)
    Xt = np.loadtxt("./test/input.tmp", ndmin=2)

# read dimensions
Ni, ni = X.shape  # input size
No, no = Y.shape  # output size
if (Ni >0 and No >0 and Ni != No):
    sys.exit("Input dimension not matching output dimension! Aborting...")

# NN design 
nhl = 4  # hidden layers size
learning_rate = 0.00005
tol = 1e-2;

## prints
print("\n\nBuilding a single layer NN")
print("hidden layer size:\t",nhl)
print("input\t\t size:\t\t", ni, "\t\t samples:", Ni)
print("output\t\t size:\t\t", no, "\t\t samples:", No)

# TF graph
# input data
x = tf.placeholder(tf.float32, shape=(None,ni)) # ?
y = tf.placeholder(tf.float32, shape=(None,no)) # ?

# weight
w1 = tf.Variable(tf.random_normal([ni, nhl],stddev=0.1, dtype=tf.float32))
w2 = tf.Variable(tf.random_normal([nhl, nhl],stddev=0.1, dtype=tf.float32))
w3 = tf.Variable(tf.random_normal([nhl, nhl],stddev=0.1, dtype=tf.float32))
w4 = tf.Variable(tf.random_normal([nhl, nhl],stddev=0.1, dtype=tf.float32))
w5 = tf.Variable(tf.random_normal([nhl, nhl],stddev=0.1, dtype=tf.float32))
wlast = tf.Variable(tf.random_normal([nhl, no],stddev=0.1, dtype=tf.float32))

# ouputs
y1 = tf.nn.relu(tf.matmul(x,w1)); # Ni x ni x ni x nhl = Ni x nhl
y2 = tf.nn.relu(tf.matmul(y1,w2)); # Ni x nhl x nhl x nhl = Ni x nhl
y3 = tf.nn.relu(tf.matmul(y2,w3)); # Ni x nhl x nhl x nhl = Ni x nhl
y4 = tf.nn.relu(tf.matmul(y3,w4)); # Ni x nhl x nhl x nhl = Ni x nhl
y5 = tf.nn.relu(tf.matmul(y4,w5)); # Ni x nhl x nhl x nhl = Ni x nhl
yhat = tf.matmul(y3,wlast); # Ni x nhl x nhl x no = Ni x no

# error
loss = tf.norm(yhat - Y, ord='euclidean') # mean square error formula
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=y)) # loss formula
train = tf.train.AdamOptimizer(learning_rate).minimize(loss) # train formula
#train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) # train formula

err = 1000.;

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("\n\nTraining NN ...")
    while sqrt(err) > tol:
        sess.run(train, feed_dict={x:X, y:Y}) # run the training
        err = sess.run(loss, feed_dict={x:X, y:Y});
        print ("error:",sqrt(err))
    Yt = sess.run(yhat, feed_dict={x:Xt})
    print("\n===== test ======\n")
    print("input\n",Xt)
    print("\noutput\n",Yt)
    
        
