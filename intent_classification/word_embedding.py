import tensorflow as tf
import numpy as np
import math

class WordEmbedding(object):
	def __init__(self,
                input_data,
                test_data,
                vocab_size,
                embedding_size,
                context_window,
                context_size):
		self.input_data = input_data
		self.test_data = test_data
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.context_size = context_size # how many words from context windows to be picked as contexts
		self.context_window = context_window  #context_window word context_window
		self.batch_size = 100
		self.iterations = 60000
		self.input_line_index = 0
		self.num_sampled = 64
		self.word_index = context_window

		self.validation_size = self.context_size*30
		self.lookup = self.train_embedding()

	def train_embedding(self):

		#prepare train validation and test validation data
		valid_train_x, valid_train_y = self.prepare_validation_pairs(self.input_data)
		valid_test_x, valid_test_y = self.prepare_validation_pairs(self.test_data)
		valid_train_words = tf.constant(valid_train_x, dtype=tf.int32)
		valid_train_contexts = tf.constant(valid_train_y, dtype=tf.int32)
		valid_test_words = tf.constant(valid_test_x, dtype=tf.int32)
		valid_test_contexts = tf.constant(valid_test_y, dtype=tf.int32)

		embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
		softmax_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                      stddev=1.0 / math.sqrt(self.embedding_size)))
		softmax_biases = tf.Variable(tf.zeros([self.vocab_size]))

		train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
		train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

		embed = tf.nn.embedding_lookup(embeddings, train_inputs)

		#Because number of class types are too many[vocab size], we use sampled_softmax_loss
		#which is a under estimation of actual softmax loss
		loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, train_labels,embed,
                                self.num_sampled, self.vocab_size))
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

		#prepare for evaluation
		#word embeddings for word and their context should be near.
		#ie., distance between them should be less
		train_word_embeddings = tf.nn.embedding_lookup(embeddings, valid_train_words)
		train_context_embeddings = tf.nn.embedding_lookup(embeddings, valid_train_contexts)
		test_word_embeddings = tf.nn.embedding_lookup(embeddings, valid_test_words)
		test_context_embeddings = tf.nn.embedding_lookup(embeddings, valid_test_contexts)
		train_distance = tf.sqrt(tf.reduce_sum(tf.square(train_word_embeddings - train_context_embeddings)))
		test_distance = tf.sqrt(tf.reduce_sum(tf.square(test_word_embeddings - test_context_embeddings)))
		
		#actual training of word embeddings goes here
		print "Word Embedding :"
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		for i in range(self.iterations):
			batch_xs, batch_ys = self.get_next_batch()
			sess.run([optimizer, loss], feed_dict={train_inputs: batch_xs, train_labels: batch_ys})
			if i%1000 == 0:
				print "train error : " + str(train_distance.eval()) + " test error : " + str(test_distance.eval())

		return embeddings.eval()

	def get_embedding(self, data):
		return self.lookup[data]

	# Given either train or test data, generates validation_size pairs
	def prepare_validation_pairs(self, data):
		assert self.validation_size%self.context_size == 0
		assert self.context_size <= 2*self.context_window
		input_word = np.ndarray((self.validation_size), dtype=np.int32)
		output_context = np.ndarray((self.validation_size, 1), dtype=np.int32)
		counter = 0
		while counter < self.validation_size/self.context_size:
			line_index = np.random.randint(len(data), size=1)
			line = data[line_index[0]]
			if (len(line) - 2*self.context_window) <= 0:
				continue
			input_word_index = np.random.randint(len(line) - 2*self.context_window, size=1)
			input_word_index = input_word_index[0] + self.context_window
			context_indices = np.random.randint(2*self.context_window, size = self.context_size)
			i = 0
			for context_index in context_indices:
				input_word[counter] = line[input_word_index]
				if context_index < self.context_window:
					output_context[counter] = line[input_word_index - context_index - 1]
				else:
					output_context[counter] = line[input_word_index + context_index - self.context_window +1]
				i+=1
				counter+=1

		return (input_word, output_context)


	def get_next_batch(self):
		#when each word is considered as a input word, it will have context_size outputs.
		#Hence we repeat the loop for batch_size/context_size times
		#Following assert statements ensures proper boundaries
		assert self.batch_size%self.context_size == 0
		assert self.context_size <= 2*self.context_window

		input_word = np.ndarray((self.batch_size), dtype=np.int32)
		output_context = np.ndarray((self.batch_size, 1), dtype=np.int32)
		counter = 0
		while counter < self.batch_size/self.context_size:
			if(self.word_index + self.context_window < len(self.input_data[self.input_line_index])):
				# for each eligible word, randomly select context_size words from a valid context window
				input_words_list = self.input_data[self.input_line_index]
				context_indices = np.random.randint(2*self.context_window, size = self.context_size)
				i = 0
				for context_index in context_indices:
					input_word[counter] = input_words_list[self.word_index]
					if context_index < self.context_window:
						output_context[counter] = input_words_list[self.word_index - context_index - 1]
					else:
						output_context[counter] = input_words_list[self.word_index + context_index - self.context_window +1]
					i+=1
					counter+=1
				self.word_index+=1
			else:
				self.input_line_index = (self.input_line_index+1)%len(self.input_data)
				self.word_index = self.context_window
		
		return (input_word, output_context)