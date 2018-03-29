import tensorflow as tf
import numpy as np

num_songs = 10
batch_size = num_songs
input_time_intervals = 16
output_time_intervals = 15

# Do, Re, Mi, Fa, So, La, Si
num_pitches = 7

learning_rate = 0.001
num_lstm_units = 100
num_lstm_layers = 2
num_epochs = 10
num_steps_in_epoch = 1000
thought_vec_dim = 10

graph = tf.Graph()

with graph.as_default():
    user_input = tf.placeholder(tf.float32, shape=[None, None, num_pitches])
    user_input_lengths = tf.placeholder(tf.int32, shape=[None])
    generator_input = tf.placeholder(tf.float32, shape=[None, None, num_pitches])
    generator_input_lengths = tf.placeholder(tf.int32, shape=[None])
    generator_expected_output = tf.placeholder(tf.float32, shape=[None, None, num_pitches])

    model_input = tf.contrib.layers.fully_connected(
        user_input, num_outputs=thought_vec_dim, activation_fn=tf.nn.sigmoid, scope="input_embedding"
    )
    model_gen_input = tf.contrib.layers.fully_connected(
        generator_input, num_outputs=thought_vec_dim, activation_fn=tf.nn.sigmoid, reuse=True, scope="input_embedding"
    )
    print("model input shape:", model_input.shape)
    print("model generator input shape:", model_gen_input.shape)

    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_lstm_units) for _ in range(num_lstm_layers)])
    my_batch_size = tf.shape(user_input)[1]
    init_state = lstm_cell.zero_state(my_batch_size, dtype=tf.float32)

    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        lstm_cell, model_input, sequence_length=user_input_lengths, initial_state=init_state, time_major=True
    )

    # For prediction
    prediction = tf.Variable(
        tf.constant(0.0, shape=[output_time_intervals, num_pitches]), dtype=tf.float32, trainable=False
    )

    current_state = encoder_state
    current_input = encoder_output[user_input_lengths[0] - 1]
    for i in range(output_time_intervals):
        prediction[i] = lstm_cell()
        pass

    # For training
    decoder_output, decoder_state = tf.nn.dynamic_rnn(
        lstm_cell, model_gen_input, sequence_length=generator_input_lengths, initial_state=encoder_state, time_major=True
    )

    final_output = tf.contrib.layers.fully_connected(
        decoder_output, num_outputs=num_pitches, activation_fn=tf.nn.sigmoid, scope="output_projection"
    )
    print("final output shape:", final_output.shape)

    # cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=generator_expected_output)
    TINY = 1e-6
    outputs = generator_expected_output
    predicted_outputs = final_output
    cross_ent = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
    loss = tf.reduce_mean(cross_ent)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

myMusicLib = [
    [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ]
]

encoder_input_batch = [[myMusicLib[bat][t] for bat in range(batch_size)] for t in range(input_time_intervals)]
encoder_input_lengths = [input_time_intervals for _ in range(batch_size)]
decoder_input_batch = [
    [myMusicLib[bat][t + input_time_intervals] for bat in range(batch_size)] for t in range(output_time_intervals)
]
decoder_expected_output = [
    [myMusicLib[bat][t + input_time_intervals + 1] for bat in range(batch_size)] for t in range(output_time_intervals)
]
decoder_input_lengths = [output_time_intervals for _ in range(batch_size)]

test_music = [
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
]
test_encoder_input_batch = [[test_music[t]] for t in range(input_time_intervals)]
test_encoder_input_lengths = [input_time_intervals]

np.set_printoptions(suppress=True, threshold=np.nan)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        sum_loss = 0.0
        for step in range(num_steps_in_epoch):
            _, current_loss, model_out = sess.run(
                [optimizer, loss, final_output],
                feed_dict={
                    user_input: encoder_input_batch,
                    user_input_lengths: encoder_input_lengths,
                    generator_input: decoder_input_batch,
                    generator_input_lengths: decoder_input_lengths,
                    generator_expected_output: decoder_expected_output
                }
            )
            sum_loss += current_loss
        avg_loss = sum_loss / num_steps_in_epoch

        print("expected_out:", decoder_expected_output)
        print("model output:", model_out)
        print("epoch:", epoch + 1)
        print("loss:", avg_loss)
    pass
