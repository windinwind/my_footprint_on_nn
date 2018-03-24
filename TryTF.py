import tensorflow as tf
import numpy as np

"""
1-bit binary adder inplemented by fully-connected neural network.
"""

INPUT_WIDTH = 2
HIDDEN_WIDTH = 4
OUTPUT_WIDTH = 2
BATCH_SIZE = 4

print("starting")

graph = tf.Graph()

# Define the graph
with graph.as_default():
    # Input data and expected output
    in_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_WIDTH])
    expected_out = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_WIDTH])

    # Weights and Bias values between input layer and hidden layer
    Wih = tf.Variable(tf.truncated_normal([INPUT_WIDTH, HIDDEN_WIDTH], stddev=0.1), dtype=tf.float32)
    Bih = tf.Variable(tf.constant(0.1, shape=[HIDDEN_WIDTH]))

    # Weights and Bias values between hidden layer and output layer
    Who = tf.Variable(tf.truncated_normal([HIDDEN_WIDTH, OUTPUT_WIDTH], stddev=0.1), dtype=tf.float32)
    Bho = tf.Variable(tf.constant(0.1, shape=[OUTPUT_WIDTH]))

    # For debug
    hidden_out = tf.nn.relu_layer(in_data, Wih, Bih)
    
    # Final output
    final_out = tf.nn.relu_layer(hidden_out, Who, Bho)

    # Calculate loss and use Adam optimizer to train the model
    loss = tf.nn.l2_loss(expected_out - final_out)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

# Define the session
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    # Create training input data and expected output data
    input = list()
    input.append([0.0, 0.0])
    input.append([0.0, 1.0])
    input.append([1.0, 0.0])
    input.append([1.0, 1.0])

    output = list()
    output.append([0.0, 0.0])
    output.append([0.0, 1.0])
    output.append([0.0, 1.0])
    output.append([1.0, 0.0])

    np.set_printoptions(suppress=True)
    # Train the model
    for i in range(2001):
        _, fin_out, l = sess.run([optimizer, final_out, loss],
                                 feed_dict={in_data: input, expected_out: output})
        if i % 1000 is 0:
            print("loss:", l, "Final out:\n", fin_out)
