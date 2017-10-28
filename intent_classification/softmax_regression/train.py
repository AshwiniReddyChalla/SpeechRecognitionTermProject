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
tf.app.flags.DEFINE_integer("iterations", 5000, "number of iterations")
tf.app.flags.DEFINE_integer("embedding_size", 25, "size of embedding")

def train():
	atis = atis_data.AtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size, FLAGS.embedding_size)
	
	number_of_labels = atis.get_number_of_labels()
  	x = tf.placeholder(tf.float32, [None, FLAGS.max_in_seq_len*FLAGS.embedding_size])
	W = tf.Variable(tf.zeros([FLAGS.embedding_size*FLAGS.max_in_seq_len, number_of_labels]))
	b = tf.Variable(tf.zeros([number_of_labels]))
	y_ = tf.placeholder(tf.float32, [None, number_of_labels])

	y = tf.matmul(x, W)+b
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	for _ in range(FLAGS.iterations):
		batch_xs, batch_ys = atis.get_next_batch(FLAGS.batch_size)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	test_x, test_y = atis.get_test_data()
	print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))



def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()