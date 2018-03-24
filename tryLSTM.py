import tensorflow as tf
import numpy as np
import random as rd

"""
Calculate the modulo of a sequence of random number using LSTM.
"""

batch_size = 5
num_range = 10
max_time = 7

graph = tf.Graph()

# Define a graph
with graph.as_default():
    # Inputs and expected outputs
    in_data = tf.placeholder(tf.float32, [max_time, batch_size, num_range])
    expected_out = tf.placeholder(tf.float32, [max_time, batch_size, num_range])

    # LSTM cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100, state_is_tuple=True)
    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, in_data, initial_state=initial_state, dtype=tf.float32, time_major=True)

    # LSTM output.
    # This is the collection of LSTM outputs in each time step, and the shape is [max_time, batch_size, num_range]
    # where final_out[t, batch] is the one hot encoding of the output number
    final_out = tf.contrib.layers.linear(outputs, num_outputs=num_range, activation_fn=tf.nn.sigmoid)

    # Calculate loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_out, labels=expected_out)
    loss = tf.reduce_mean(loss)

    # Use Adam Optimizer to train the model
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

np.set_printoptions(suppress=True)

# Define the session (training)
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1000):
        sum_loss = 0.0
        for step in range(1000):
            # Generate random input sequence and calculate the output
            training_input_number = list()
            training_input = np.empty((max_time, batch_size, num_range))
            training_output = np.empty((max_time, batch_size, num_range))
            for i in range(batch_size):
                in_seq = [rd.randrange(num_range) for _ in range(max_time)]
                training_input_number.append(in_seq)
                modulo = [(sum(in_seq[:(xx + 1)])) % num_range for xx in range(max_time)]

                one_hot_in = [[0 for _ in range(num_range)] for _ in range(max_time)]
                one_hot_out = [[0 for _ in range(num_range)] for _ in range(max_time)]
                for j in range(max_time):
                    one_hot_in[j][in_seq[j]] = 1
                    one_hot_out[j][modulo[j]] = 1
                    training_input[j, i] = one_hot_in[j]
                    training_output[j, i] = one_hot_out[j]

            # Train the model
            _, current_loss, model_out = sess.run([optimizer, loss, final_out], feed_dict={in_data: training_input, expected_out: training_output})
            sum_loss += current_loss
        avg_loss = sum_loss / 100.0
        print("loss:", avg_loss)
        print("model input:", training_input_number)

        # Convert one hot prediction to the actual number
        actual_out = list()
        for i in range(batch_size):
            prediction_one_hot = [model_out[xx, i] for xx in range(max_time)]
            prediction = []
            for xx in range(max_time):
                max_val = list(prediction_one_hot[xx]).index(max(prediction_one_hot[xx]))
                prediction.append(max_val)
            actual_out.append(prediction)
        # Print the actual prediction
        print("model output:", actual_out)
