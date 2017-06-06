import tensorflow as tf 

tf.reset_default_graph()

with tf.name_scope("input"):
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    result = tf.add(a,b, name="result")


writer = tf.summary.FileWriter("/output/training_logs/log1", tf.get_default_graph(), flush_secs=1)
writer.close()


tf.reset_default_graph()

with tf.name_scope("input"):
    a = tf.constant([1.0, 2.0], name="a")
    b = tf.constant([2.0, 3.0], name="b")
    
with tf.name_scope("output"):
	result = tf.add(a,b, name="result")


writer = tf.summary.FileWriter("/output/training_logs/log2", tf.get_default_graph(), flush_secs=1)
writer.close()