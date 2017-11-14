import tensorflow as tf
import numpy as np
import sys
sys.path.append("../../intent_classification")
import atis_data 
class TransferAtisData(object):
	def __init__(self,
	            data_folder,
	            vocab_size,
	            max_in_seq_len,
	            max_data_size,
	            no_of_base_intents = 8):
		if no_of_base_intents > 14 or no_of_base_intents < 1:
			raise ValueError("Number of base intents should be between [1, 14]. %d", no_of_base_intents)
		self.max_in_seq_len = max_in_seq_len
		self.max_data_size = max_data_size
		self.no_of_base_intents = no_of_base_intents

		old_atis = atis_data.AtisData(data_folder, vocab_size, max_in_seq_len, max_data_size)
		data = self.prepare_data_for_transfer_learning(old_atis)
		self.old_atis = old_atis
		self.vocab_size = self.old_atis.vocab_size

		self.labels_train = self.old_atis.labels_train = data[0]
		self.in_seq_train = self.old_atis.in_seq_train = data[1]
		self.labels_transfer_train = data[2]
		self.in_seq_transfer_train = data[3]

		self.labels_test = self.old_atis.labels_test = data[4]
		self.in_seq_test = self.old_atis.in_seq_test = data[5]
		self.labels_transfer_test = data[6]
		self.in_seq_transfer_test = data[7]

		self.labels_valid = self.old_atis.labels_valid = data[8]
		self.in_seq_valid = self.old_atis.in_seq_valid = data[9]
		self.labels_transfer_valid = data[10]
		self.in_seq_transfer_valid = data[11]

		self.transfer_batch_index = 0

		self.old_atis.no_of_class_labels = self.get_number_of_base_labels()
		self.total_no_of_class_labels = self.get_number_of_transfer_labels() + self.get_number_of_base_labels()

	def prepare_data_for_transfer_learning(self, old_atis):
		result = []
		labels = [old_atis.labels_train, old_atis.labels_test, old_atis.labels_dev]
		in_seq = [old_atis.in_seq_train, old_atis.in_seq_test, old_atis.in_seq_dev]

		for counter in range(3):
			labels_base =[]
			in_seq_base = []
			labels_transfer = []
			in_seq_transfer = []

			for i in range(len(labels[counter])):
				if labels[counter][i][0] < self.no_of_base_intents:
					labels_base.append(labels[counter][i])
					in_seq_base.append(in_seq[counter][i])
				else:
					labels_transfer.append(labels[counter][i])
					in_seq_transfer.append(in_seq[counter][i])

			result.append(labels_base)
			result.append(in_seq_base)
			result.append(labels_transfer)
			result.append(in_seq_transfer)

		return result

	def get_number_of_transfer_labels(self):
		unique_labels = []
		[unique_labels.extend(label) for label in self.labels_transfer_train] 
		return len(set(unique_labels)) 

	def get_number_of_base_labels(self):
		return self.old_atis.get_number_of_labels()

	def get_vocab_size(self):
		return self.old_atis.get_vocab_size()

	def get_next_batch_for_base_learning(self, batch_size, one_hot_y=True, one_hot_x=False):
		return self.old_atis.get_next_batch(batch_size, one_hot_y, one_hot_x)

	def get_base_test_data(self, one_hot_y=True, one_hot_x=False):
		return self.old_atis.get_test_data(one_hot_y, one_hot_x)

	def get_base_train_data(self, one_hot_y=True, one_hot_x=False):
		return self.old_atis.get_train_data(one_hot_y, one_hot_x)

	def get_transfer_test_data(self, one_hot_y=True, one_hot_x=False):
		base_test_X, base_test_Y = self.old_atis.get_test_data(one_hot_y, one_hot_x, self.total_no_of_class_labels)

		#get transfer data
		transfer_test_X, transfer_test_Y = self.old_atis.get_input_output_data(
					self.in_seq_transfer_test, 
					self.labels_transfer_test, 
					one_hot_y, one_hot_x, self.total_no_of_class_labels)

		return (np.concatenate((base_test_X, transfer_test_X), axis = 0) , 
			np.concatenate((base_test_Y, transfer_test_Y), axis = 0))

	def get_transfer_train_data(self, one_hot_y=True, one_hot_x=False):
		base_train_X, base_train_Y = self.old_atis.get_train_data(one_hot_y, one_hot_x, self.total_no_of_class_labels)

		#get transfer data
		transfer_train_X, transfer_train_Y = self.old_atis.get_input_output_data(
					self.in_seq_transfer_train, 
					self.labels_transfer_train, 
					one_hot_y, one_hot_x, self.total_no_of_class_labels)

		return (np.concatenate((base_train_X, transfer_train_X), axis = 0) , 
			np.concatenate((base_train_Y, transfer_train_Y), axis = 0))

	def get_next_batch_for_transfer_learning(self, batch_size, one_hot_y=True, one_hot_x=False):
		transfer_train_X, transfer_train_Y = self.old_atis.get_input_output_data(
					self.in_seq_transfer_train, 
					self.labels_transfer_train, 
					one_hot_y, one_hot_x, self.total_no_of_class_labels)

		train_input = []
		train_labels = []
		counter = 0
		train_data_size = len(self.in_seq_train)
		last_index = self.transfer_batch_index + batch_size
		if last_index <= train_data_size:
			train_input = self.in_seq_train[self.transfer_batch_index : last_index]
			train_labels = self.labels_train[self.transfer_batch_index : last_index]
			if last_index == train_data_size:
				self.transfer_batch_index = 0
			else:
				self.transfer_batch_index = last_index
		else:
			train_input = self.in_seq_train[self.transfer_batch_index :]
			train_labels = self.labels_train[self.transfer_batch_index :]
			self.transfer_batch_index = batch_size - len(train_input) 
			train_input.extend(self.in_seq_train[: self.transfer_batch_index])
			train_labels.extend(self.labels_train[: self.transfer_batch_index])

		base_train_X, base_train_Y = self.old_atis.get_input_output_data(train_input, train_labels, one_hot_y, one_hot_x, self.total_no_of_class_labels)	
		return (np.concatenate((base_train_X, transfer_train_X), axis = 0) , 
			np.concatenate((base_train_Y, transfer_train_Y), axis = 0))



