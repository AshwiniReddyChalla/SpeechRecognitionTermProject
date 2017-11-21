from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np

import data_helper
import word_embedding

class AtisData(object):
  def __init__(self,
                data_folder,
                vocab_size,
                max_in_seq_len,
                max_data_size,
                embed_size = -1):

    self.max_in_seq_len = max_in_seq_len
    self.max_data_size = max_data_size
    self.batch_index = 0;

    # Get words in the form of numbers :  0 - vocabsize ; 0 for unk
    data = data_helper.get_tokenized_data(data_folder, vocab_size)
    tokenized_in_seq_path_train, tokenized_label_path_train = data[0]
    tokenized_in_seq_path_valid, tokenized_label_path_valid = data[1]
    tokenized_in_seq_path_test, tokenized_label_path_test = data[2]
    in_vocab_path, label_vocab_path = data[3]

    # read the data in form of numbers into memory
    self.in_seq_train = self.read_data_into_memory(tokenized_in_seq_path_train);
    self.labels_train = self.read_data_into_memory(tokenized_label_path_train);
    self.in_seq_valid = self.read_data_into_memory(tokenized_in_seq_path_valid);
    self.labels_valid = self.read_data_into_memory(tokenized_label_path_valid);
    self.in_seq_test = self.read_data_into_memory(tokenized_in_seq_path_test);
    self.labels_test = self.read_data_into_memory(tokenized_label_path_test);
    self.in_vocab_path = in_vocab_path;
    self.vocab_size = self.get_vocab_size();

    if(len(self.in_seq_train) != len(self.labels_train)) :
       raise ValueError("Number of train labels != Number of train inputs : %d != %d",len(self.in_seq_train), len(self.labels_train))

    if(len(self.in_seq_valid) != len(self.labels_valid)) :
       raise ValueError("Number of valid labels != Number of valid inputs : %d != %d",len(self.in_seq_valid), len(self.labels_valid))

    if(len(self.in_seq_test) != len(self.labels_test)) :
       raise ValueError("Number of test labels != Number of test inputs : %d != %d",len(self.in_seq_test), len(self.labels_test))

    self.no_of_class_labels = self.get_number_of_labels();
    self.embed_size = embed_size
    self.word_embedding = None
    
    # construct word embedding if required by the model
    if embed_size > 0:
      self.word_embedding = word_embedding.WordEmbedding(
                        self.in_seq_train,
                        self.in_seq_test,
                        self.vocab_size,
                        embed_size,
                        2, #context_window
                        1 #context_size
                        )

  def get_vocab_size(self):
    with tf.gfile.GFile(self.in_vocab_path, mode="r") as source_file:
      lines = source_file.readlines()
      return len(lines)

  def get_number_of_labels(self):
    unique_labels = []
    [unique_labels.extend(label) for label in self.labels_train] 
    return len(set(unique_labels)) 
  
  def read_data_into_memory(self, file_path):
    data = []
    with tf.gfile.GFile(file_path, mode="r") as source_file:
      lines = source_file.readlines()
      if len(lines) > self.max_data_size:
        lines = lines[:self.max_data_size]
      for line in lines:
        line = line.strip();
        data.append([int(x) for x in line.split()]);
    return data

  def set_max_in_seq_len(self, max_in_seq_len):
    self.max_in_seq_len = max_in_seq_len

  # pad the data to get all sentences with particular sentence length
  def get_padded_data(self, input):
    padded_inputs = []

    for i in range(len(input)):
      data = input[i]
      if len(data) < self.max_in_seq_len:
        pad = [data_helper.PAD_ID for _ in range(self.max_in_seq_len - len(data))]
        data.extend(pad);
      else:
        data = data[:self.max_in_seq_len]
      padded_inputs.append(np.array(data, np.int32))

    return np.array(padded_inputs)

  def get_one_hot_encoded_labels(self, labels, no_of_class_labels = None):
    if no_of_class_labels is None:
      no_of_class_labels = self.no_of_class_labels
    one_hot_encoded_labels = []
    for label in labels:
      data = [0 for _ in range(no_of_class_labels)]
      data[label[0]] = 1
      one_hot_encoded_labels.append(np.array(data, np.int32))

    return np.array(one_hot_encoded_labels)

  def get_one_hot_encoded_input_data(self, data):
    one_hot_encoded_input_data = np.ndarray((len(data), self.max_in_seq_len*self.vocab_size), dtype=np.int32)
    for i in range(len(data)):
      o_h = []
      line = data[i]
      for j in range(self.max_in_seq_len):
        one_hot = [0 for _ in range(self.vocab_size)]
        if line[j] > 0:
          one_hot[line[j] - 1] = 1
        o_h.extend(one_hot)
      one_hot_encoded_input_data[i] = o_h
    return one_hot_encoded_input_data

  def get_embedded_input_data(self, data):
    if not self.word_embedding:
      return data
    embedded_data = np.ndarray((len(data), self.max_in_seq_len, self.embed_size), dtype=np.int32)
    for i in range(len(data)):
      line = data[i]
      for j in range(self.max_in_seq_len):
        embedded_data[i,j,:] = self.word_embedding.get_embedding(line[j])
    
    return embedded_data

  def get_test_data(self, one_hot_y=True, one_hot_x=False, no_of_class_labels=None):
    if no_of_class_labels is None:
      no_of_class_labels = self.no_of_class_labels
    return self.get_input_output_data(self.in_seq_test, self.labels_test, one_hot_y, one_hot_x, no_of_class_labels)

  def get_valid_data(self, one_hot_y=True, one_hot_x=False, no_of_class_labels=None):
    if no_of_class_labels is None:
      no_of_class_labels = self.no_of_class_labels
    return self.get_input_output_data(self.in_seq_valid, self.labels_valid, one_hot_y, one_hot_x, no_of_class_labels)

  def get_train_data(self, one_hot_y=True, one_hot_x=False, no_of_class_labels=None):
    if no_of_class_labels is None:
      no_of_class_labels = self.no_of_class_labels
    return self.get_input_output_data(self.in_seq_train, self.labels_train, one_hot_y, one_hot_x, no_of_class_labels)

  def get_input_output_data(self, in_seq, labels, one_hot_y=True, one_hot_x=False, no_of_class_labels=None):
    if no_of_class_labels is None:
      no_of_class_labels = self.no_of_class_labels
    padded_data = self.get_padded_data(in_seq)
    if one_hot_x :
      X = self.get_one_hot_encoded_input_data(padded_data)
    else:
      X = self.get_embedded_input_data(padded_data)

    Y = labels
    if one_hot_y:
      Y = self.get_one_hot_encoded_labels(Y, no_of_class_labels)
    else:
      Y = np.array(Y)
    return (X,Y)

    
  #Gets batch size of training data each time looping over training data.
  # Returns one-hot-encoded Y, [one-hot encoded or embedded vectors of data]
  def get_next_batch(self, batch_size, one_hot_y=True, one_hot_x=False):
    train_input = []
    train_labels = []
    counter = 0
    train_data_size = len(self.in_seq_train)
    last_index = self.batch_index + batch_size
    if last_index <= train_data_size:
      train_input = self.in_seq_train[self.batch_index : last_index]
      train_labels = self.labels_train[self.batch_index : last_index]
      if last_index == train_data_size:
        self.batch_index = 0
      else:
        self.batch_index = last_index
    else:
        train_input = self.in_seq_train[self.batch_index :]
        train_labels = self.labels_train[self.batch_index :]

        #epoch completed - shuffle data
        self.in_seq_train, self.labels_train = shuffle(self.in_seq_train, self.labels_train)
       
        #give remaining data
        self.batch_index = batch_size - len(train_input) 
        train_input.extend(self.in_seq_train[: self.batch_index])
        train_labels.extend(self.labels_train[: self.batch_index])

    return self.get_input_output_data(train_input, train_labels, one_hot_y, one_hot_x)