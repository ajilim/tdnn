import numpy as np
import tensorflow as tf


input_dim=4

hidden_units=[3,2,2]

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

train_data=np.loadtxt("data.txt",dtype=float)
train_target=train_data[:,-1]
train_feature=train_data[:,0:-1]
test_data=np.loadtxt("data.txt",dtype=float)
test_target=test_data[:,-1]
test_feature=test_data[:,0:-1]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=hidden_units,
                                          model_dir="./",
                                          n_classes=3)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x":train_feature},
      y=train_target,
      num_epochs=None,
      shuffle=True)

print(train_target)
print(train_feature.shape)
classifier.train(input_fn=train_input_fn, steps=2000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_feature},
      y=test_target,
      num_epochs=1,
      shuffle=False)


accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
