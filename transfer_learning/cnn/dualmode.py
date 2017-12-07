import tensorflow as tf
import math
import sys
import os.path
import numpy as np
import shutil
from plot_accuracy import plot_accuracy
from plot_accuracy import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def train(atis, max_in_seq_len, embedding_size, iterations, batch_size, base_training = True, restore_from_ckpt = True, save_to_ckpt = False, freeze_model = False, confusion_matrix_file_name = None):
	num_filters = 16
	filter_sizes = [2, 3, 4, 5, 6, 7]
	num_filters_total = num_filters * len(filter_sizes)
	normal_initializer = tf.random_normal_initializer(stddev=0.1)

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

		dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		embedding = tf.get_variable("embedding",
											shape=[atis.vocab_size, embedding_size],
											initializer=normal_initializer, trainable = not freeze_model)
		variables_to_save["embedding"] = embedding
		embedded_words = tf.nn.embedding_lookup(embedding, train_x )#[None,sentence_length,embed_size]

		embedded_expanded = tf.expand_dims(embedded_words, -1)
		# [None,sentence_length, embed_size, 1)
		if base_training:
			W_projection_name = "W_base"
			b_projection_name = "b_base"
		else:
			W_projection_name = "W_projection"
			b_projection_name = "b_projection"
			no_of_base_labels = atis.get_number_of_base_labels()
			W_base = tf.get_variable("W_base",
				shape=[num_filters_total, no_of_base_labels]) #[feature _size,label_size]
			b_base = tf.get_variable("b_base",shape=[1, no_of_base_labels]) #[label_size] 

		W_projection = tf.get_variable(W_projection_name,
				shape=[num_filters_total, num_classes],
				initializer=normal_initializer) #[feature _size,label_size]
		b_projection = tf.get_variable(b_projection_name,shape=[1, num_classes]) #[label_size] 

		pooled_outputs = []
		for i,filter_size in enumerate(filter_sizes):
			with tf.name_scope("convolution-pooling-%s" %filter_size):
				filter = tf.get_variable("filter-%s"%filter_size,
							[filter_size, embedding_size,1,num_filters],
							initializer=normal_initializer, trainable = not freeze_model)
				#[filter_size, embed_size ,1, num_filters]
				variables_to_save["filter-%s"%filter_size] = filter

				conv = tf.nn.conv2d(embedded_expanded, filter, strides=[1,1,1,1], padding="VALID",name="conv") 
				#shape:[batch_size, sequence_length - filter_size + 1, 1, num_filters]

				b = tf.get_variable("b-%s"%filter_size, [num_filters], trainable = not freeze_model)
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
		params = tf.global_variables()
		global_step = tf.Variable(0, trainable=False)
		starter_learning_rate = 0.005
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           50, 0.9, staircase=False)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		grads_and_vars = optimizer.compute_gradients(loss, params)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		saveable_params = []
		for param in params:
			if 'projection' not in param.name:
				saveable_params.append(param)
		saver = tf.train.Saver(saveable_params, reshape=True)
		config=tf.ConfigProto()

		valid_acc_new_list = []
		valid_acc_whole_list = []
		with tf.Session(graph = g) as sess:
			tf.global_variables_initializer().run()
			if not base_training and restore_from_ckpt:
				print "restoring ....."
				saver.restore(sess, "./ckpt/cnn.ckpt")
				no_of_transfer_labels = atis.get_number_of_transfer_labels()
				W_init = tf.concat([W_base, tf.zeros([num_filters_total, no_of_transfer_labels], tf.float32)], axis=1)
				b_init = tf.concat([b_base, tf.zeros([1, no_of_transfer_labels], tf.float32)], axis=1)
				
				W_projection_assign = tf.assign(W_projection, W_init)
				b_projection_assign = tf.assign(b_projection, b_init)
				sess.run(W_projection_assign)
				sess.run(b_projection_assign)
				#W_projection.initializer.run()
				#b_projection.initializer.run()
				print "done restoring"

			for i in range(1, iterations+1):
				if base_training:
					batch_xs, batch_ys = atis.get_next_batch_for_base_learning(batch_size)
				else:
					batch_xs, batch_ys = atis.get_next_batch_for_transfer_learning(batch_size)
				sess.run([train_op, global_step, loss, accuracy], feed_dict={train_x: batch_xs, train_y: batch_ys, dropout_keep_prob:0.6})
				
				if base_training:
					valid_x, valid_y = atis.get_base_valid_data()
					x, y = atis.get_base_train_data()
					valid_acc = sess.run(accuracy, feed_dict={train_x: valid_x, train_y: valid_y, dropout_keep_prob:1.0})
					train_acc = sess.run(accuracy, feed_dict={train_x: x, train_y: y, dropout_keep_prob:1.0})

					valid_acc_whole_list.append(valid_acc)
					#there is no new data in base training hence giving training acc instead
					valid_acc_new_list.append(train_acc)
				else:
					valid_x, valid_y = atis.get_whole_test_data()
					valid_new_x, valid_new_y = atis.get_only_new_test_data()
					
					valid_acc_whole = sess.run(accuracy, feed_dict={train_x: valid_x, train_y: valid_y, dropout_keep_prob:1.0})
					valid_acc_new = sess.run(accuracy, feed_dict={train_x: valid_new_x, train_y: valid_new_y, dropout_keep_prob:1.0})
					
					valid_acc_whole_list.append(valid_acc_whole)
					valid_acc_new_list.append(valid_acc_new)

			if base_training or save_to_ckpt:
				if os.path.exists("./ckpt/"):
					shutil.rmtree("./ckpt/")
				save_path = saver.save(sess, "./ckpt/cnn.ckpt")
				print "model saved in " + save_path

			#print test accuracy
			if base_training:
				test_x, test_y = atis.get_base_test_data()
				whole_test_accuracy = sess.run(accuracy, feed_dict={train_x: test_x, train_y: test_y, dropout_keep_prob:1.0})
				print("test accuracy: " + str(whole_test_accuracy))
				new_test_accuracy = 0
			else:
				test_x, test_y = atis.get_whole_valid_data()
				new_test_x, new_test_y = atis.get_only_new_valid_data()
				whole_test_accuracy = sess.run(accuracy, feed_dict={train_x: test_x, train_y: test_y, dropout_keep_prob:1.0})
				new_test_accuracy = sess.run(accuracy, feed_dict={train_x: new_test_x, train_y: new_test_y, dropout_keep_prob:1.0})
				print("test accuracy: " + str(whole_test_accuracy) + 
				" new test accuracy: " + str(new_test_accuracy))
				
			if confusion_matrix_file_name is not None:
				#plot confusion matrix
				y_actual = tf.argmax(test_y, 1)
				y_actual = y_actual.eval()
				y_pred = sess.run(predictions, feed_dict={train_x: test_x, train_y: test_y, dropout_keep_prob:1.0})
				cm_matrix = confusion_matrix(y_actual, y_pred)

				plot_confusion_matrix(cm_matrix, classes=[str(i) for i in range(1, num_classes+1)], 
					file_name=confusion_matrix_file_name, normalize=True, title='Confusion matrix')


			return valid_acc_new_list, valid_acc_whole_list, new_test_accuracy, whole_test_accuracy
