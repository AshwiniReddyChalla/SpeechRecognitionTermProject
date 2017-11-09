import tensorflow as tf
import math
import sys
sys.path.append("../")
import atis_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", '../../ATIS', "Data directory")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_string("max_in_seq_len", 30, "max in seq length")
tf.app.flags.DEFINE_integer("max_data_size", 10000, "max training data size")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("iterations", 10000, "number of iterations")
tf.app.flags.DEFINE_integer("embedding_size", 300, "size of embedding")

def train():
	atis = atis_data.AtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size, FLAGS.embedding_size)

	# define parameters of cnn
	num_filters = 32
	filter_sizes = [2, 3, 4]
	num_filters_total = num_filters * len(filter_sizes)
	dropout_keep_prob = 0.4
	normal_initializer = tf.random_normal_initializer(stddev=0.1)

	#get data
	num_classes = atis.get_number_of_labels()
	l2_loss = tf.constant(0.0)

	train_x = tf.placeholder(tf.float32, [None, FLAGS.max_in_seq_len, FLAGS.embedding_size])
	# [None,sentence_length,embed_size]
	train_y = tf.placeholder(tf.float32, [None, num_classes])
	embedded_expanded = tf.expand_dims(train_x, -1)
	# [None,sentence_length, feature_size, 1)
	
	W_projection = tf.get_variable("W_projection",
			shape=[num_filters_total, num_classes],
			initializer=normal_initializer) #[feature _size,label_size]
	b_projection = tf.get_variable("b_projection",shape=[num_classes]) #[label_size] 

	pooled_outputs = []
	for i,filter_size in enumerate(filter_sizes):
		with tf.name_scope("convolution-pooling-%s" %filter_size):
			filter = tf.get_variable("filter-%s"%filter_size,
						[filter_size,FLAGS.embedding_size,1,num_filters],
						initializer=normal_initializer)
			#[filter_size, embed_size ,1, num_filters]

			conv = tf.nn.conv2d(embedded_expanded, filter, strides=[1,1,1,1], padding="VALID",name="conv") 
			#shape:[batch_size, sequence_length - filter_size + 1, 1, num_filters]

			b = tf.get_variable("b-%s"%filter_size, [num_filters])
			h = tf.nn.relu(tf.nn.bias_add(conv,b),"relu") 
			#shape:[batch_size, sequence_length - filter_size + 1, 1, num_filters]

			pooled = tf.nn.max_pool(h, 
									ksize=[1,FLAGS.max_in_seq_len-filter_size+1,1,1], 
									strides=[1,1,1,1], 
									padding='VALID',
									name="pool")
			#shape:[batch_size, 1, 1, num_filters]

			pooled_outputs.append(pooled)

	h_pool=tf.concat(pooled_outputs, 3) #shape:[batch_size, 1, 1, num_filters_total]
	h_pool_flat=tf.reshape(h_pool,[-1,num_filters_total])

	h_drop = tf.nn.dropout(h_pool_flat, keep_prob = dropout_keep_prob) 
	#[None, num_filters_total]

	logits = tf.matmul(h_drop, W_projection) + b_projection
	losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_y)
	loss = tf.reduce_mean(losses)  

	predictions = tf.argmax(logits, 1)
	correct_predictions = tf.equal(predictions, tf.argmax(train_y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	global_step = tf.Variable(0, name="global_step", trainable=False)
	optimizer = tf.train.AdamOptimizer(1e-3)
	grads_and_vars = optimizer.compute_gradients(loss)
	train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

	config=tf.ConfigProto()
	with tf.Session(config=config) as sess:
		tf.global_variables_initializer().run()
		for i in range(FLAGS.iterations):
			batch_xs, batch_ys = atis.get_next_batch(FLAGS.batch_size)
			sess.run([train_op, global_step, loss, accuracy], feed_dict={train_x: batch_xs, train_y: batch_ys})
			if i%100 == 0:
				test_x, test_y = atis.get_test_data()
				print("test accuracy: " + str(sess.run(accuracy, feed_dict={train_x: test_x, train_y: test_y})) + 
					" train accuracy : " + str(sess.run(accuracy, feed_dict={train_x: batch_xs, train_y: batch_ys})))

def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()