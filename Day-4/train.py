import tensorflow as tf

# prepare the dataset loader
from caltechfaces.loader import get_loader
maxhw = 160 # resize images larger than 160x160 pixels to that size
load_batch = get_loader(maxhw)

# `x` is a batch of input images (each of size (maxhw, maxhw) pixels with 3 channels)
x = tf.placeholder(tf.float32, [None, maxhw, maxhw, 3])
# `y` is a batch of ground-truth per-pixel labels (same size as `x`)
y = tf.placeholder(tf.float32, [None, maxhw, maxhw, 3])

from models import *
pred, is_training = build_model(x, 3) # `x` is input, `3` specifies the number of output channels

# prepare the loss function
thr = 0.1
loss = tf.nn.relu( 0.5*(y - pred)**2 - 0.5*thr**2 )
# loss = tf.nn.relu(tf.abs(y - pred) - thr)
loss = tf.reduce_sum(tf.reduce_mean(loss, axis=[1,2,3]))

global_step = tf.Variable(0, trainable=False)
num_iters = 2048
lr = tf.train.polynomial_decay(5e-4, global_step, num_iters,
                               end_learning_rate=0, power=1.0)
# we will use Adam to learn the model
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	# step = tf.train.AdamOptimizer(5e-4).minimize(loss)
	step = tf.train.AdamOptimizer(lr).minimize(loss)

#
saver = tf.train.Saver()

#
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for k in range(num_iters):
	#
	# X, Y = load_batch(100)
	X, Y = load_batch(32)
	l, _ = sess.run([loss, step], feed_dict={x: X, y: Y, is_training: True})
	print('* loss on batch %d: %.7f' % (k, l))

save_dir = 'save'
print('Save dir:', save_dir)
import os
os.system('mkdir -p ' + save_dir)
save_path = saver.save(sess, save_dir + '/model')
print("Model saved to `%s`" % save_path)

print('* done ...')