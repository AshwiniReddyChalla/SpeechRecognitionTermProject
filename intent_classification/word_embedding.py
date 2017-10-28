import tensorflow as tf
import numpy as np
import math

class WordEmbedding(object):
	def __init__(self,
                input_data,
                vocab_size,
                embedding_size,
                context_window,
                context_size):
		self.input_data = input_data
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.context_size = context_size # how many words from context windows to be picked as contexts
		self.context_window = context_window  #context_window word context_window
		self.batch_size = 500
		self.iterations = 1000
		self.input_line_index = 0
		self.num_sampled = 32
		self.word_index = context_window

		self.lookup = self.train_embedding()

	def train_embedding(self):
		embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
		nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                      stddev=1.0 / math.sqrt(self.embedding_size)))
		nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
		train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
		train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, self.context_size])
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		embed = tf.nn.embedding_lookup(normalized_embeddings, train_inputs)
		loss = tf.reduce_mean(
  				 tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=self.num_sampled,
                 num_classes=self.vocab_size))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		for _ in range(self.iterations):
			batch_xs, batch_ys = self.get_next_batch()
			sess.run([optimizer, loss], feed_dict={train_inputs: batch_xs, train_labels: batch_ys})

		return normalized_embeddings.eval()

	def get_embedding(self, data):
		return self.lookup[data]

	def get_next_batch(self):
		input_word = np.ndarray((self.batch_size), dtype=np.int32)
		output_context = np.ndarray((self.batch_size, self.context_size), dtype=np.int32)
		counter = 0
		while counter < self.batch_size:
			if(self.word_index + self.context_window < len(self.input_data[self.input_line_index])):
				input_words_list = self.input_data[self.input_line_index]
				input_word[counter] = input_words_list[self.word_index]
				context_indices = np.random.randint(2*self.context_window, size = self.context_size)
				i = 0
				for context_index in context_indices:
					if context_index < self.context_window:
						output_context[counter, i] = input_words_list[self.word_index - context_index - 1]
					else:
						output_context[counter, i] = input_words_list[self.word_index + context_index - self.context_window +1]
					i+=1
				self.word_index+=1
				counter+=1
			else:
				self.input_line_index = (self.input_line_index+1)%len(self.input_data)
				self.word_index = self.context_window
		return (input_word, output_context)