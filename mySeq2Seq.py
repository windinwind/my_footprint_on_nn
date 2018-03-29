import tensorflow as tf
import numpy as np
import random as rd

# Hyper Parameters
batch_size = 1
input_num_time_intervals = 4  # Input max_time
output_num_time_intervals = 10  # Output max_time
num_pitches = 31
learning_rate = 0.001
num_lstm_units = 512
num_lstm_layers = 2
num_steps_in_epoch = 1000

graph = tf.Graph()


def sos_token(max_number):
    return [0 for _ in range(max_number)]


def num_to_one_hot(number, max_number):
    if number is -1:
        return sos_token(max_number)
    result = [0 for _ in range(max_number)]
    result[number % max_number] = 1
    return result


def generate_encoder_id_sequence(sequence_length, max_number):
    return [rd.randrange(0, max_number) for _ in range(sequence_length)]


def generate_encoder_id_batch(bat_size, sequence_length, max_number):
    return [generate_encoder_id_sequence(sequence_length, max_number) for _ in range(bat_size)]


def generate_decoder_id_sequence(encoder_id_seq, sequence_length, max_number):
    current_number = sum(encoder_id_seq) % max_number
    result = []
    for _ in range(sequence_length):
        result.append(current_number)
        current_number = (current_number * 2) % max_number
    return result


def generate_decoder_id_batch(encoder_id_batch, bat_size, sequence_length, max_number):
    return [generate_decoder_id_sequence(encoder_id_batch[bat], sequence_length, max_number) for bat in range(bat_size)]


def generate_one_hot_batch(id_sequences, bat_size, sequence_length, max_number):
    # shape: (max_time, batch_size, one_hot_length)
    return [[num_to_one_hot(id_sequences[bat][seq], max_number) for bat in range(bat_size)] for seq in range(sequence_length)]


use_different_cells = True

# Define the graph
with graph.as_default():
    encoder_input = tf.placeholder(tf.float32, shape=[input_num_time_intervals, None, num_pitches])

    # out1, out2, ..., outN, <EOS>
    expected_sequence_output = tf.placeholder(tf.float32, shape=[output_num_time_intervals, None, num_pitches])

    # <SOS>, out1, out2, ..., outN
    decoder_training_input = tf.placeholder(tf.float32, shape=[output_num_time_intervals, None, num_pitches])
    decoder_batch_lengths = tf.placeholder(tf.int32, shape=[batch_size])

    input_seq_length = tf.placeholder(tf.int32, shape=[batch_size])

    if (use_different_cells):
        with tf.variable_scope("encoder"):
            # Encoder Cell
            def get_encoder_cell(num_units, num_layers):
                return tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units) for _ in range(num_layers)])
            encoder_cell = get_encoder_cell(num_lstm_units, num_lstm_layers)
            init_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)

            # Run the encoder to obtain the encoder state
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input, initial_state=init_state, time_major=True)

        with tf.variable_scope("decoder"):
            def get_decoder_cell(num_units):
                return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_decoder_cell(num_lstm_units) for _ in range(num_lstm_layers)]
            )
            """
            decoder_outputs, decoder_state = tf.nn.dynamic_rnn(
                decoder_cell, decoder_training_input, initial_state=encoder_state, time_major=True
            )"""

            helper = tf.contrib.seq2seq.TrainingHelper(decoder_training_input, decoder_batch_lengths, time_major=True)
    else:
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_lstm_units) for _ in range(num_lstm_layers)])
        decoder_cell = encoder_cell

        init_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_input, initial_state=init_state, time_major=True)
        decoder_outputs, decoder_state = tf.nn.dynamic_rnn(
            decoder_cell, decoder_training_input, initial_state=encoder_state, time_major=True
        )

    projection_layer = tf.layers.Dense(num_pitches, use_bias=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, encoder_state, output_layer=projection_layer
    )

    final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
    final_output = final_outputs.rnn_output

    cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=expected_sequence_output)
    loss = tf.reduce_mean(cross_ent)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

np.set_printoptions(suppress=True)
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1000):
        sum_loss = 0.0
        for step in range(num_steps_in_epoch):
            # Generate random input sequence and calculate the output
            encoder_id_batch = generate_encoder_id_batch(batch_size, input_num_time_intervals, num_pitches)
            decoder_id_batch = generate_decoder_id_batch(encoder_id_batch, batch_size, output_num_time_intervals, num_pitches)

            decoder_input_id_batch = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                decoder_input_id_batch[i].append(-1)
                for j in range(output_num_time_intervals - 1):
                    decoder_input_id_batch[i].append(decoder_id_batch[i][j])

            encoder_one_hot_input = generate_one_hot_batch(encoder_id_batch, batch_size, input_num_time_intervals, num_pitches)

            decoder_one_hot_input = generate_one_hot_batch(decoder_input_id_batch, batch_size, output_num_time_intervals, num_pitches)
            decoder_one_hot_output = generate_one_hot_batch(decoder_id_batch, batch_size, output_num_time_intervals, num_pitches)

            # Train the model
            _, current_loss, model_out = sess.run(
                [optimizer, loss, final_output],
                feed_dict={encoder_input: encoder_one_hot_input,
                           decoder_training_input: decoder_one_hot_input,
                           expected_sequence_output: decoder_one_hot_output,
                           decoder_batch_lengths: [output_num_time_intervals for _ in range(batch_size)]}
            )
            sum_loss += current_loss
        avg_loss = sum_loss / num_steps_in_epoch
        print("loss:", avg_loss)
        print("model input:", encoder_id_batch)
        actual_out = []
        """for i in range(batch_size):
            prediction_one_hot = [model_out[xx, i] for xx in range(output_num_time_intervals)]
            prediction = []
            for xx in range(output_num_time_intervals):
                max_val = list(prediction_one_hot[xx]).index(max(prediction_one_hot[xx]))
                prediction.append(max_val)
            actual_out.append(prediction)

        # Print the actual prediction
        print("model output:", prediction)"""
        for i in range(batch_size):
            prediction = []
            for j in range(output_num_time_intervals):
                max_val = list(model_out[i][j]).index(max(model_out[i][j]))
                prediction.append(max_val)
            actual_out.append(prediction)

        print("expected_out:", decoder_id_batch)
        print("model output:", actual_out)
