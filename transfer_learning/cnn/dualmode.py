import tensorflow as tf
import math
import sys
import os.path

def train(atis, max_in_seq_len, embedding_size, iterations, batch_size, base_training = True):
	num_filters = 32
	filter_sizes = [2, 3, 4]
	num_filters_total = num_filters * len(filter_sizes)
	dropout_keep_prob = 0.4
	normal_initializer = tf.random_normal_initializer(stddev=0.1)

	if base_training:
		if os.path.exists('./ckpt'):
			return
		scope_name = "base"
	else:
		scope_name = "transfer"

	g = tf.Graph()
	with g.as_default(): 
		variables_to_save = {}

		#get data
		if base_training:
			num_classes = atis.get_number_of_base_labels()
		else:
			num_classes = atis.get_number_of_base_labels() + atis.get_number_of_transfer_labels()

		print "number of  classes are : "
		print num_classes

		train_x = tf.placeholder(tf.int32, [None, max_in_seq_len])
		# [None,sentence_length]
		train_y = tf.placeholder(tf.float32, [None, num_classes])

		embedding = tf.get_variable("embedding",
											shape=[atis.vocab_size, embedding_size],
											initializer=normal_initializer)
		variables_to_save["embedding"] = embedding
		embedded_words = tf.nn.embedding_lookup(embedding, train_x )#[None,sentence_length,embed_size]

		embedded_expanded = tf.expand_dims(embedded_words, -1)
		# [None,sentence_length, embed_size, 1)
		
		W_projection = tf.get_variable("W_projection",
				shape=[num_filters_total, num_classes],
				initializer=normal_initializer) #[feature _size,label_size]
		b_projection = tf.get_variable("b_projection",shape=[num_classes]) #[label_size] 

		pooled_outputs = []
		for i,filter_size in enumerate(filter_sizes):
			with tf.name_scope("convolution-pooling-%s" %filter_size):
				filter = tf.get_variable("filter-%s"%filter_size,
							[filter_size, embedding_size,1,num_filters],
							initializer=normal_initializer)
				#[filter_size, embed_size ,1, num_filters]
				variables_to_save["filter-%s"%filter_size] = filter

				conv = tf.nn.conv2d(embedded_expanded, filter, strides=[1,1,1,1], padding="VALID",name="conv") 
				#shape:[batch_size, sequence_length - filter_size + 1, 1, num_filters]

				b = tf.get_variable("b-%s"%filter_size, [num_filters])
				variables_to_save["b-%s"%filter_size] = b
				h = tf.nn.relu(tf.nn.bias_add(conv,b),"relu") 
				#shape:[batch_size, sequence_length - filter_size + 1, 1, num_filters]

				pooled = tf.nn.max_pool(h, 
										ksize=[1, max_in_seq_len-filter_size+1,1,1], 
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
		params = tf.trainable_variables()
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(loss, params)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		saver = tf.train.Saver(variables_to_save)
		config=tf.ConfigProto()
		with tf.Session(graph = g) as sess:
			if base_training:
				tf.global_variables_initializer().run()
			else:
				print "restoring .....\n\n\n"
				saver.restore(sess, "./ckpt/cnn.ckpt")
				W_projection.initializer.run()
				b_projection.initializer.run()
				print "done...\n\n\n"

			for i in range(iterations):
				if base_training:
					batch_xs, batch_ys = atis.get_next_batch_for_base_learning(batch_size)
				else:
					batch_xs, batch_ys = atis.get_next_batch_for_transfer_learning(batch_size)
				sess.run([train_op, global_step, loss, accuracy], feed_dict={train_x: batch_xs, train_y: batch_ys})
				if (i < 100 and i %10 == 0) or i%100 == 0:
					if base_training:
						test_x, test_y = atis.get_base_test_data()
						x, y = atis.get_base_train_data()
					else:
						test_x, test_y = atis.get_transfer_test_data()
						x, y = atis.get_transfer_train_data()
					print("test accuracy: " + str(sess.run(accuracy, feed_dict={train_x: test_x, train_y: test_y})) + 
						" train accuracy : " + str(sess.run(accuracy, feed_dict={train_x: x, train_y: y})))

			if base_training:
				save_path = tf.train.Saver(variables_to_save).save(sess, "./ckpt/cnn.ckpt")
				print "model saved in " + save_path
