import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

input_dim=784
layer1_dim=10
#relu-renorm-layer 650dim

layer2_dim=100
layer3_dim=100
layer4_dim=100
layer5_dim=10

learning_rate=0.3

# train_data=np.loadtxt("g_data.txt",dtype=float).astype("float")
# train_target=train_data[:,-layer5_dim:]
# train_feature=train_data[:,0:-layer5_dim]
# test_data=np.loadtxt("g_data.txt",dtype=float).astype("float")
# test_target=test_data[:,-layer5_dim:]
# test_feature=test_data[:,0:-layer5_dim]

#print(train_target.shape)

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

w1=tf.Variable(tf.random_normal([input_dim,layer1_dim]))
w2=tf.Variable(tf.random_normal([layer1_dim,layer2_dim]))
w3=tf.Variable(tf.random_normal([layer2_dim,layer3_dim]))
w4=tf.Variable(tf.random_normal([layer3_dim,layer4_dim]))
w5=tf.Variable(tf.random_normal([layer4_dim,layer5_dim]))

b1=tf.Variable(tf.random_normal([1,layer1_dim]))
b2=tf.Variable(tf.random_normal([1,layer2_dim]))
b3=tf.Variable(tf.random_normal([1,layer3_dim]))
b4=tf.Variable(tf.random_normal([1,layer4_dim]))
b5=tf.Variable(tf.random_normal([1,layer5_dim]))

layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w3), b3))
layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, w4), b4))
layer_5 = tf.nn.softmax(tf.add(tf.matmul(layer_4, w5), b5))

#loss=tf.reduce_mean(tf.square(layer_1-y))
loss=tf.reduce_mean(-tf.reduce_sum(y * tf.log(layer_5), reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    # we'll make 5000 gradient descent iteration
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _,err =session.run([train_op,loss], feed_dict={x: batch_xs, y: batch_ys})
        print(err)
    #print(session.run(layer_5,feed_dict={x:test_feature,y:test_target}))
    correct_prediction = tf.equal(tf.argmax(layer_5, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))