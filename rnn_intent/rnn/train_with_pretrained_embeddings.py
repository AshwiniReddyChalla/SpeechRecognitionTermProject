# LSTM and CNN for sequence classification in the IMDB dataset
import tensorflow as tf
import numpy
import math
import pickle
import sys
sys.path.append("../")
import atis_data

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


# fix random seed for reproducibility
numpy.random.seed(7)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", '../../NEW_ATIS', "Data directory")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_string("max_in_seq_len", 30, "max in seq length")
tf.app.flags.DEFINE_integer("max_data_size", 10000, "max training data size")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("iterations", 10000, "number of iterations")
tf.app.flags.DEFINE_integer("embedding_size", 100, "size of embedding")

# load the dataset
atis = atis_data.AtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size, FLAGS.embedding_size)
X_train, y_train = atis.get_train_data()
X_test, y_test = atis.get_test_data()
y_train = to_categorical(y_train, num_classes=18)
y_test = to_categorical(y_test, num_classes=18)

# truncate and pad input sequences
in_vocab_size = 5000
embedding_size = 100
max_in_seq_len = numpy.shape(X_train)[1]

X_train = sequence.pad_sequences(X_train, maxlen=max_in_seq_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_in_seq_len)

print numpy.shape(X_train)
print numpy.shape(y_train)
print numpy.shape(X_test)
print numpy.shape(y_test)

# create the model
model = Sequential()
model.add(Embedding(5000, 300, input_length=max_in_seq_len))
model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(18, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

'''
# create the model
model = Sequential()
model.add(Embedding(in_vocab_size, embedding_size, input_length=max_in_seq_len))
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(18, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, batch_size=100)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
'''
