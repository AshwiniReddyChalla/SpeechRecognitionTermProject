import tensorflow as tf
import math
import sys
import os.path
import numpy as np
import shutil
from tensorflow.contrib import rnn
from plot_accuracy import plot_accuracy
from plot_accuracy import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

def train(atis, max_in_seq_len, embedding_size, iterations, batch_size, base_training = True, restore_from_ckpt = True, save_to_ckpt = False, freeze_model = False, confusion_matrix_file_name = None):
	
	no_of_fw_cells = 40
	no_of_bw_cells = 40
	num_features_total = no_of_fw_cells + no_of_bw_cells
	normal_initializer = tf.random_normal_initializer(stddev=0.1)

	g = tf.Graph()
	with g.as_default(): 

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
		embedded_words = tf.nn.embedding_lookup(embedding, train_x )#[None,sentence_length,embed_size]

		if base_training:
			W_projection_name = "W_base"
			b_projection_name = "b_base"
		else:
			W_projection_name = "W_projection"
			b_projection_name = "b_projection"
			no_of_base_labels = atis.get_number_of_base_labels()
			W_base = tf.get_variable("W_base",
				shape=[num_features_total, no_of_base_labels]) #[feature _size,label_size]
			b_base = tf.get_variable("b_base",shape=[1, no_of_base_labels]) #[label_size] 

		W_projection = tf.get_variable(W_projection_name,
				shape=[num_features_total, num_classes],
				initializer=normal_initializer) #[feature _size,label_size]
		b_projection = tf.get_variable(b_projection_name,shape=[1, num_classes]) #[label_size] 

		#define forward and backward cells
		lstm_fw_cell=rnn.DropoutWrapper(rnn.BasicLSTMCell(no_of_fw_cells),
				output_keep_prob=dropout_keep_prob)
		lstm_bw_cell=rnn.DropoutWrapper(rnn.BasicLSTMCell(no_of_bw_cells),
				output_keep_prob=dropout_keep_prob)

		#create dynamic bidirectional recurrent neural network
		outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,
				embedded_words,dtype=tf.float32) 
		#outputs = [[batch_size,sequence_length,no_of_fw_cells], [batch_size,sequence_length,no_of_bw_cells] ]
		output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,num_features_total]
		features = tf.reduce_mean(output_rnn,axis=1) #[batch_size,num_features_total]

		logits = tf.matmul(features, W_projection) + b_projection
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_y)
		loss = tf.reduce_mean(losses)  

		predictions = tf.argmax(logits, 1)
		correct_predictions = tf.equal(predictions, tf.argmax(train_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
		params = tf.trainable_variables()
		global_step = tf.Variable(0, trainable=False)
		starter_learning_rate = 0.01
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           50, 0.9, staircase=False)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		trainable_params = [param for param in params if (not freeze_model) or 'lstm' not in param.name]
		grads_and_vars = optimizer.compute_gradients(loss, trainable_params)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		saveable_params = []
		for param in params:
			if 'projection' not in param.name:
				saveable_params.append(param)
		saver = tf.train.Saver(saveable_params, reshape=True)

		print "Total number of params = " + str(count_number_trainable_params(not freeze_model))

		config=tf.ConfigProto()

		valid_acc_new_list = []
		valid_acc_whole_list = []
		with tf.Session(graph = g) as sess:
			tf.global_variables_initializer().run()
			if not base_training and restore_from_ckpt:
				print "restoring ....."
				saver.restore(sess, "./ckpt/cnn.ckpt")
				no_of_transfer_labels = atis.get_number_of_transfer_labels()
				W_init = tf.concat([W_base, tf.zeros([num_features_total, no_of_transfer_labels], tf.float32)], axis=1)
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

def count_number_trainable_params(count_lstm = True):
	'''
	Counts the number of trainable variables.
	'''
	tot_nb_params = 0
	for trainable_variable in tf.trainable_variables():
		if not count_lstm and 'lstm' in trainable_variable.name:
			continue
		shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
		current_nb_params = get_nb_params_shape(shape)
		tot_nb_params = tot_nb_params + current_nb_params
	return tot_nb_params

def get_nb_params_shape(shape):
	'''
	Computes the total number of params for a given shap.
	Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
	'''
	nb_params = 1
	for dim in shape:
		nb_params = nb_params*int(dim)
	return nb_params 
