from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, [1,None, 784])
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
#y = tf.nn.softmax(tf.matmul(x, W) + b)

h1=tf.layers.conv1d(x,100,1,use_bias=1)
h2=tf.layers.conv1d(tf.nn.sigmoid(h1),10,1,use_bias=1)
y=tf.nn.softmax(h2)
y=tf.squeeze(y)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_xs=np.reshape(batch_xs,(1,100,784))
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
testx=mnist.test.images
testy=mnist.test.labels

print(sess.run(accuracy, feed_dict={x: np.reshape(mnist.test.images,(1,10000,784)), y_: mnist.test.labels}))