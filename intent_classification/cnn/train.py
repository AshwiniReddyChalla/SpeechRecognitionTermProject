import tensorflow as tf

import sys
sys.path.append("../")
import atis_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", '../../ATIS', "Data directory")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_string("max_in_seq_len", 20, "max in seq length")
tf.app.flags.DEFINE_integer("max_data_size", 4000, "max training data size")
tf.app.flags.DEFINE_integer("batch_size", 500, "batch size")
tf.app.flags.DEFINE_integer("iterations", 1000, "number of iterations")
tf.app.flags.DEFINE_integer("embedding_size", 25, "size of embedding")

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def train():
	atis = atis_data.AtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size)

	number_of_labels = atis.get_number_of_labels()
	vocab_size = atis.get_vocab_size();

	x_train = tf.placeholder(tf.int32, shape=[None, FLAGS.max_in_seq_len])
	y_train = tf.placeholder(tf.int32, shape=[None, number_of_labels])

	#embedding layer 
	embedding_weights = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0))
	embedded_inputs = tf.nn.embedding_lookup(embedding_weights, x_train)
	embedded_inputs_expanded = tf.expand_dims(embedded_inputs, -1)

	#make number and size of convolutional layers as flag
	#convolutional layer
	
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(embedded_inputs_expanded, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([5 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 5*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, number_of_labels])
	b_fc2 = bias_variable([number_of_labels])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(
	    tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_train, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(FLAGS.iterations):
			batch_xs, batch_ys = atis.get_next_batch(FLAGS.batch_size)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x_train: batch_xs, y_train: batch_ys, keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict={x_train: batch_xs, y_train: batch_ys, keep_prob: 0.5})
		test_x, test_y = atis.get_test_data()
		print(sess.run(accuracy, feed_dict={x_train: test_x, y_train: test_y, keep_prob: 0.5}))

def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()
	