import tensorflow as tf 

tf.reset_default_graph()

def inference(input_tensor, reuse=False):
	with tf.variable_scope("layer1", reuse=reuse):
		weights = tf.get_variable("weights", [784,500], 
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		biases = tf.get_variable("biases", [500],
			initializer=tf.constant_initializer(0.0))
		print biases
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
	with tf.variable_scope("layer2", reuse=reuse):
		weights = tf.get_variable("weights", [500, 10],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		biases = tf.get_variable("biases",[10],
			initializer=tf.constant_initializer(0.0))
		print biases
		layer2 = tf.nn.relu(tf.matmul(layer1, weights)+biases)


x = tf.placeholder(tf.float32, [None, 784], name="x_input")
y = inference(x)

writer = tf.summary.FileWriter("/output/training_logs/log3")
writer.close()

