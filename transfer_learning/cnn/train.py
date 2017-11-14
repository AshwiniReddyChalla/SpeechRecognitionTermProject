import tensorflow as tf
import math
import sys
sys.path.append("../")
import transfer_atis_data
import dualmode
import os.path

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", '../../NEW_ATIS', "Data directory")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_string("max_in_seq_len", 30, "max in seq length")
tf.app.flags.DEFINE_integer("max_data_size", 10000, "max training data size")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("iterations", 50, "number of iterations")
tf.app.flags.DEFINE_integer("embedding_size", 300, "size of embedding")
tf.app.flags.DEFINE_integer("no_of_base_intents", 8, "number of base intents")


# define parameters of cnn
num_filters = 32
filter_sizes = [2, 3, 4]
num_filters_total = num_filters * len(filter_sizes)
dropout_keep_prob = 0.4
normal_initializer = tf.random_normal_initializer(stddev=0.1)


def transfer_train():
	atis = transfer_atis_data.TransferAtisData(FLAGS.data_dir, FLAGS.in_vocab_size,
			 FLAGS.max_in_seq_len, FLAGS.max_data_size, FLAGS.no_of_base_intents)
	dualmode.train(atis, FLAGS.max_in_seq_len, FLAGS.embedding_size, FLAGS.iterations, FLAGS.batch_size, True)
	dualmode.train(atis, FLAGS.max_in_seq_len, FLAGS.embedding_size, FLAGS.iterations, FLAGS.batch_size, False)

	
def main(_):
    transfer_train()

if __name__ == "__main__":
  transfer_train()