from __future__ import print_function

import tensorflow as tf

logs_path = "/output/training_logs"

x = tf.placeholder(tf.float32, shape=(2,1), name='x')

# 3*2矩阵
w1 = tf.Variable([[0.2,0.3],[0.1, -0.5],[0.4, 0.2]], name='w1')

# 1*3 矩阵
w2 = tf.Variable([[0.6, 0.1, -0.2]], name='w2')

a = tf.matmul(w1, x, name='a')
y = tf.matmul(w2, a, name='y')

summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(y, feed_dict={x: [[0.7],[0.9]]}))
	summary_writer.close()

