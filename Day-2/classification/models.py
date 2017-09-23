import tensorflow as tf

def make_affine(x):
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	logits = tf.matmul(x, W) + b
	return logits

def make_convnet(x):
	# reshape input to image
	x = tf.reshape(x, [-1, 28, 28, 1])
	# first conv block
	x = tf.layers.conv2d(x, 32, 5, padding='SAME')
	x = tf.nn.relu(x)
	x = tf.layers.max_pooling2d(x, 2, 2)
	# second conv block
	x = tf.layers.conv2d(x, 64, 5, padding='SAME')
	x = tf.nn.relu(x)
	x = tf.layers.max_pooling2d(x, 2, 2)
	#
	x = tf.reshape(x, [-1, 7*7*64])
	logits = tf.layers.dense(x, 10)
	return logits