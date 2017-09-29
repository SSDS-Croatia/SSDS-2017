import tensorflow as tf

# need this placeholder for bach norm
is_training = tf.placeholder(tf.bool)
# batch norm params
bn_params = {
  # Decay for the moving averages.
  'momentum': 0.9,
  # epsilon to prevent 0s in variance.
  'epsilon': 1e-5,
  # fused must be false if BN is frozen
  'fused': True,
  'training': is_training
}

def conv(x, num_maps, k=3):
  # x = tf.layers.conv2d(x, num_maps, k,
  #   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), padding='same')
  # x = tf.layers.conv2d(x, num_maps, k, padding='same')

  x = tf.layers.conv2d(x, num_maps, k, use_bias=False,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), padding='same')
  x = tf.layers.batch_normalization(x, training=is_training)
  return tf.nn.relu(x)

def pool(x):
  return tf.layers.max_pooling2d(x, 2, 2, 'same')

def upsample(x, skip, num_maps):
  skip_size = skip.get_shape().as_list()[1:3]
  x = tf.image.resize_bilinear(x, skip_size)
  x = tf.concat([x, skip], 3)
  return conv(x, num_maps)


def block(x, size, name):
  with tf.name_scope(name):
    x = conv(x, size)
  return x

def build_model(x, num_classes):
  print(x)
  input_size = x.get_shape().as_list()[1:3]
  block_sizes = [64, 64, 64, 64]
  skip_layers = []
  x = conv(x, 32)
  for i, size in enumerate(block_sizes):
    x = pool(x)
    x = conv(x, size)
    x = conv(x, size)
    if i < len(block_sizes) - 1:
      skip_layers.append(x)

  # # 36 without
  for i, skip in reversed(list(enumerate(skip_layers))):
    print(i, x, '\n', skip)
    x = upsample(x, skip, block_sizes[i])
  print('final: ', x)
  x = tf.layers.conv2d(x, num_classes, 1, padding='same')
  x = tf.image.resize_bilinear(x, input_size, name='upsample_logits')
  return tf.nn.sigmoid(x), is_training