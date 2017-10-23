import tensorflow as tf
import numpy as np

output_dim=3
input_dim=1
learning_rate=0.0001
train_data=np.loadtxt("g_data.txt",dtype=float).astype("float")
train_target=train_data[:,-output_dim:]
train_feature=train_data[:,0:-output_dim]
test_data=np.loadtxt("g_data_test.txt",dtype=float).astype("float")
test_target=test_data[:,-output_dim:]
test_feature=test_data[:,0:-output_dim]

#以下用于reshape
train_feature_row,train_feature_col=train_feature.shape
train_feature=np.reshape(train_feature,(1,train_feature_row,train_feature_col))

train_target_row,train_target_col=train_target.shape
train_target=np.reshape(train_target,(1,train_target_row,train_target_col))

test_feature_row,test_feature_col=test_feature.shape
test_feature=np.reshape(test_feature,(1,test_feature_row,test_feature_col))

test_target_row,test_target_col=test_target.shape
test_target=np.reshape(test_target,(1,test_target_row,test_target_col))

input=tf.placeholder(tf.float32,[1,None,input_dim])
y=tf.placeholder(tf.float32,[1,None,output_dim])
#这样做是因为在conv1d中，stride是一维的，因此只能将行数变为1，在列数上stride，同时，深度为特征的维数
imaginput=np.random.normal(size=(1,1500,input_dim))
#frame:11,feature:8,depth:1
h1_kernel_size = 3
h1_filters = 20
#h1_filters:output_dim
h1 = tf.layers.conv1d(input, h1_filters, h1_kernel_size)
h1_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h1))


#h1_kernel_size:一次性拿多少时间点
h2_kernel_size = 3
h2_filters=20
h2=tf.layers.conv1d(h1_relu_renorm,h2_filters,h2_kernel_size)
h2_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h2))

h3_kernel_size = 3
h3_filters=20
h3=tf.layers.conv1d(h2_relu_renorm,h3_filters,h3_kernel_size)
h3_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h3))

h4_kernel_size = 3
h4_filters=20
h4=tf.layers.conv1d(h3_relu_renorm,h4_filters,h4_kernel_size)
h4_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h4))

h5_kernel_size = 3
h5_filters=20
h5=tf.layers.conv1d(h4_relu_renorm,h5_filters,h5_kernel_size)
h5_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h5))

output_kernel_size=3
output_filter=output_dim
output=tf.layers.conv1d(h5_relu_renorm,output_filter,output_kernel_size)
output_softmax=tf.nn.softmax(output)

loss=tf.reduce_mean(tf.square(output_softmax-y))
#loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(output_softmax)))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess=tf.Session()
sess.run(tf.global_variables_initializer())



for i in range(0,80):
    feature=train_feature[:,i*300:300*(i+1),:]
    target=train_target[:,i*288:288*(i+1),:]
    feature=np.reshape(feature,(1,300,input_dim))
    target = np.reshape(target, (1, 288, output_dim))
    for j in range(0,300):
        _,a=sess.run([train_op,loss],feed_dict={input:feature,y:target})
        print(a)
i = 0
feature=test_feature[:,i*300:300*(i+1),:]
target=test_target[:,i*288:288*(i+1),:]
feature=np.reshape(feature,(1,300,input_dim))
target = np.reshape(target, (1, 288, output_dim))

#b=sess.run(output_softmax,feed_dict={input:test_feature,y:test_target})
#c=tf.argmax(tf.squeeze(output_softmax),1)

prediction=tf.equal(tf.argmax(tf.squeeze(output_softmax),1),tf.argmax(tf.squeeze(y),1))
accuracy=tf.reduce_mean(tf.cast(prediction,tf.float32))
print("prediction:")
print(sess.run(accuracy,feed_dict={input:feature,y:target}))
#print(tf.shape(output_softmax))