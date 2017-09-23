import tensorflow as tf

# every MNIST sample has two parts: an image of a handwritten digit and a corresponding label
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# `x` is a batch of input images
x = tf.placeholder(tf.float32, [None, 784])

from models import *
#logits = make_convnet(x)
logits = make_affine(x)

# prepare the loss function
labels = tf.placeholder(tf.float32, [None, 10])
loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# we will use SGD to learn the model
step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

# prepare the accuracy-computation graph
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for k in range(5000):
	batch_xs, batch_labels = mnist.train.next_batch(100)
	sess.run(step, feed_dict={x: batch_xs, labels: batch_labels})
	if k % 200 == 0:
		acc = 100*sess.run(accuracy, feed_dict={x: mnist.test.images, labels: mnist.test.labels})
		print('* iter %d: test set accuracy=%.2f %%' % (k, acc))

print('* done ...')