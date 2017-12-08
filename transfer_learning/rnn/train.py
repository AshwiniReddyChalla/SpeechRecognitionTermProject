import tensorflow as tf
import math
import sys
sys.path.append("../")
import transfer_atis_data
import dualmode
import plotly.graph_objs as go
import os.path
import numpy as np
from plot_accuracy import plot_accuracy_num_train_samples, plot_accuracy

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", '../../NEW_ATIS', "Data directory")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_string("max_in_seq_len", 30, "max in seq length")
tf.app.flags.DEFINE_integer("max_data_size", 10000, "max training data size")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("iterations", 30, "number of iterations")
tf.app.flags.DEFINE_integer("embedding_size", 300, "size of embedding")
tf.app.flags.DEFINE_integer("no_of_base_intents", 8, "number of base intents")
tf.app.flags.DEFINE_integer("total_no_of_intents", 15, "number of total intents")

def plot_test_accuracy_versus_samples():
	iterations = 30
	
	no_pretrain_new_test_accuracy = []
	no_pretrain_whole_test_accuracy = []
	pretrained_freeze_new_test_accuracy = []
	pretrained_freeze_whole_test_accuracy = []
	pretrained_no_freeze_new_test_accuracy = []
	pretrained_no_freeze_whole_test_accuracy = []
	max_samples = 35
	for i in range(1, max_samples):
		print "training with " + str(i) + "sample(s)...\n\n\n"
		_,_,new_test, whole_test = train_all_without_pretrained_model(iterations, i)
		no_pretrain_new_test_accuracy.append(new_test)
		no_pretrain_whole_test_accuracy.append(whole_test)

		_,_,new_test, whole_test = train_all_with_pretrained_frozen_model(iterations, i)
		pretrained_freeze_new_test_accuracy.append(new_test)
		pretrained_freeze_whole_test_accuracy.append(whole_test)

		_,_,new_test, whole_test = train_all_with_pretrained_not_frozen_model(iterations, i)
		pretrained_no_freeze_new_test_accuracy.append(new_test)
		pretrained_no_freeze_whole_test_accuracy.append(whole_test)

	data = []
	data.append(go.Scatter(
		y = np.array(no_pretrain_new_test_accuracy),
		x = np.array(range(1, max_samples)),
		mode = 'lines',
		name = 'without pretrained'
    ))
	data.append(go.Scatter(
		y = np.array(pretrained_freeze_new_test_accuracy),
		x = np.array(range(1, max_samples)),
		mode = 'lines',
		name = 'with pretrained frozen'
    ))
	data.append(go.Scatter(
		y = np.array(pretrained_no_freeze_new_test_accuracy),
		x = np.array(range(1, max_samples)),
		mode = 'lines',
		name = 'with pretrained not frozen'
    ))

	plot_accuracy(data, "samples", "new_test_accuracy", "new_accuracy_samples.html")

	data = []
	data.append(go.Scatter(
		y = np.array(no_pretrain_whole_test_accuracy),
		x = np.array(range(1, max_samples)),
		mode = 'lines',
		name = 'without pretrained'
	))
	data.append(go.Scatter(
		y = np.array(pretrained_freeze_whole_test_accuracy),
		x = np.array(range(1, max_samples)),
		mode = 'lines',
		name = 'with pretrained frozen'
	))
	data.append(go.Scatter(
		y = np.array(pretrained_no_freeze_whole_test_accuracy),
		x = np.array(range(1, max_samples)),
		mode = 'lines',
		name = 'with pretrained not frozen'
	))

	plot_accuracy(data, "samples", "whole_test_accuracy", "whole_accuracy_samples.html")


#plots new test accuracy and over all test accuracy versus number of iterations
# for 3 modes
# 1. without pretrained
# 2. with pretrained frozen model
# 3. with pretrained and non frozen model
# From this plot we can infer how transfer learning helps quickly learn in few iterations
def plot_test_accuracy_versus_iterations():
	iterations = 100
	
	no_pretrain_new_test_accuracy, no_pretrain_whole_test_accuracy, _, _ = train_all_without_pretrained_model(iterations, cm_file_name = './cm_without_pretrained.png')
	print "with no pretrained model done ......\n\n\n"

	perform_base_training(iterations)
	print "base training done ...\n\n\n"

	pretrained_freeze_new_test_accuracy, pretrained_freeze_whole_test_accuracy, _, _ = train_all_with_pretrained_frozen_model(iterations, cm_file_name = './cm_pretrained_frozen.png')
	print "with pretrained and freeze done ......\n\n\n"
	
	pretrained_no_freeze_new_test_accuracy, pretrained_no_freeze_whole_test_accuracy, _, _ = train_all_with_pretrained_not_frozen_model(iterations, cm_file_name = './cm_pretrained_not_frozen.png')	
	print "with pretrained and no freeze done ......\n\n\n"
	
	data = []
	data.append(go.Scatter(
		y = np.array(no_pretrain_new_test_accuracy),
		x = np.array(range(1, iterations+1)),
		mode = 'lines',
		name = 'without pretrained'
    ))
	data.append(go.Scatter(
		y = np.array(pretrained_freeze_new_test_accuracy),
		x = np.array(range(1, iterations+1)),
		mode = 'lines',
		name = 'with pretrained frozen'
    ))
	data.append(go.Scatter(
		y = np.array(pretrained_no_freeze_new_test_accuracy),
		x = np.array(range(1, iterations+1)),
		mode = 'lines',
		name = 'with pretrained not frozen'
    ))

	plot_accuracy(data, "iterations", "new_test_accuracy", "new_accuracy_iterations.html")

	data = []
	data.append(go.Scatter(
		y = np.array(no_pretrain_whole_test_accuracy),
		x = np.array(range(1, iterations+1)),
		mode = 'lines',
		name = 'without pretrained'
	))
	data.append(go.Scatter(
		y = np.array(pretrained_freeze_whole_test_accuracy),
		x = np.array(range(1, iterations+1)),
		mode = 'lines',
		name = 'with pretrained frozen'
	))
	data.append(go.Scatter(
		y = np.array(pretrained_no_freeze_whole_test_accuracy),
		x = np.array(range(1, iterations+1)),
		mode = 'lines',
		name = 'with pretrained not frozen'
	))

	plot_accuracy(data, "iterations", "whole_test_accuracy", "whole_accuracy_iterations.html")

def train_all_without_pretrained_model(iterations, num_train_samples = -1, cm_file_name = None):
	atis = transfer_atis_data.TransferAtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size, 0, 7, 8, 14, max_num_of_samples_per_new_class = num_train_samples)
	return dualmode.train(atis, FLAGS.max_in_seq_len, FLAGS.embedding_size, iterations, FLAGS.batch_size, 
					base_training = False, 
					restore_from_ckpt = False, 
					save_to_ckpt = False,
					freeze_model = False,
					confusion_matrix_file_name = cm_file_name)

def train_all_with_pretrained_frozen_model(iterations, num_train_samples = -1, cm_file_name = None):
	atis = transfer_atis_data.TransferAtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size, 0, 7, 8, 14, max_num_of_samples_per_new_class = num_train_samples)
	return dualmode.train(atis, FLAGS.max_in_seq_len, FLAGS.embedding_size, iterations, FLAGS.batch_size, 
					base_training = False, 
					restore_from_ckpt = True, 
					save_to_ckpt = False,
					freeze_model = True,
					confusion_matrix_file_name = cm_file_name)

def train_all_with_pretrained_not_frozen_model(iterations, num_train_samples = -1, cm_file_name = None):
	atis = transfer_atis_data.TransferAtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size, 0, 7, 8, 14, max_num_of_samples_per_new_class = num_train_samples)
	return dualmode.train(atis, FLAGS.max_in_seq_len, FLAGS.embedding_size, iterations, FLAGS.batch_size, 
					base_training = False, 
					restore_from_ckpt = True, 
					save_to_ckpt = False,
					freeze_model = False,
					confusion_matrix_file_name = cm_file_name)

def perform_base_training(iterations):
	atis = transfer_atis_data.TransferAtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size, 0, 7, -1, -1)
	return dualmode.train(atis, FLAGS.max_in_seq_len, FLAGS.embedding_size, iterations, FLAGS.batch_size, 
					base_training = True, 
					restore_from_ckpt = False, 
					save_to_ckpt = True,
					freeze_model = False)

if __name__ == "__main__":
  plot_test_accuracy_versus_iterations()