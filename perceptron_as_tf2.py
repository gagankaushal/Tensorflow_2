import numpy as np
import matplotlib.pyplot as plt

## EDIT number 1
######### Following code edited by Gagan so as to convert the code to Tensorflow 2 ##########
# laura added these lines to make tf 1 code compatible w/ tf 2
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
##############################################################################################

NUM_FEATURES = 2
NUM_ITER = 2000
learning_rate = 0.01

x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)  # 4x2, input
# y = np.array`([0, 0, 1, 0], np.float32)  # 4, correct output, AND operation
y = np.array([0, 1, 1, 1], np.float32)  # OR operation

y_for_plot = y # laura added this line
y = np.reshape(y, [4, 1])  # convert to 4x1

## EDIT number 2
######### Following code modified by Gagan so as to convert the code to Tensorflow 2 ##########
# X = tf.placeholder(tf.float32, shape=[4, 2])
# Y = tf.placeholder(tf.float32, shape=[4, 1])
X = tf.keras.Input(shape=(2,),batch_size=4, dtype = tf.dtypes.float32 )
Y = tf.keras.Input(shape=(1,),batch_size=4, dtype = tf.dtypes.float32)
X=x
Y=y
##############################################################################################

W = tf.Variable(tf.zeros([NUM_FEATURES, 1]), tf.dtypes.float32)
B = tf.Variable(tf.zeros([1, 1]), tf.dtypes.float32)

## EDIT number 3
######## Following code modified by Gagan so as to convert the code to Tensorflow 2 ##################
# yHat = tf.sigmoid(tf.add(tf.matmul(X, W), B))  # 4x1
# err = Y - yHat
# deltaW = tf.matmul(tf.transpose(X), err)  # have to be 2x1
# deltaB = tf.reduce_sum(err, 0)  # 4, have to 1x1. sum all the biases? yes
# W_ = W + learning_rate * deltaW
# B_ = B + learning_rate * deltaB
#
# step = tf.group(W.assign(W_), B.assign(B_))  # to update the values of weights and biases.

for k in range(NUM_ITER):

        yHat = tf.sigmoid(tf.add(tf.matmul(X, W), B))  # 4x1
        err = Y - yHat
        deltaW = tf.matmul(tf.transpose(X), err)  # have to be 2x1
        deltaB = tf.reduce_sum(err, 0)  # 4, have to 1x1. sum all the biases? yes

        W_ = W + learning_rate * deltaW
        B_ = B + learning_rate * deltaB
        tf.group(W.assign(W_), B.assign(B_))  # to update the values of weights and biases.
##############################################################################################

## EDIT number 4
######### Following code removed by Gagan so as to convert the code to Tensorflow 2 ##########
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
##############################################################################################

## EDIT number 5
################ Updated by Gagan so as to convert the code to Tensorflow 2 ##################
# W = np.squeeze(sess.run(W))
# b = np.squeeze(sess.run(B))
W = W.numpy()
b = B.numpy()
W = np.squeeze(W)
b = np.squeeze(b)
##############################################################################################

# Now plot the fitted line. We need only two points to plot the line
plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1] + 0.2)])
# print('value of b',b)
plot_y = - 1 / W[1] * (W[0] * plot_x + b)
plot_y = np.reshape(plot_y, [2, -1])
plot_y = np.squeeze(plot_y)

print('W: ' + str(W))
print('b: ' + str(b))
print('plot_y: ' + str(plot_y))

plt.scatter(x[:, 0], x[:, 1], c=y_for_plot, s=100, cmap='viridis') # laura changed this line
plt.plot(plot_x, plot_y, color='k', linewidth=2)
plt.xlim([-0.2, 1.2])
plt.ylim([-0.2, 1.25])
plt.show()
