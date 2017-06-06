import tensorflow as tf


x = tf.placeholder(tf.float32, shape=(2,1), name='x')

# 3*2
w1 = tf.Variable([[0.2, 0.3],[0.1, -0.5],[0.4, 0.2]], name = 'w1')

b1 = tf.constant([[-0.5],[0.1],[-0.1]])

# 1*3
w2 = tf.Variable([[0.6,0.1,-0.2]], name = 'w2')

b2 = tf.constant([[0.1]])

a = tf.nn.relu(tf.matmul(w1, x) +b1)
y = tf.nn.relu(tf.matmul(w2, a) + b2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(y, feed_dict={x: [[0.7], [0.9]]}))
