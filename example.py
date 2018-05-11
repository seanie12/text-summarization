import tensorflow as tf
from tensorflow.contrib import layers, seq2seq
from tensorflow.python.layers.core import Dense
import numpy as np

EOS = 0
GO = 1
state_size = 20
decoder_vocab_size = 7
encoder_input = tf.constant([[2, 2, 3, 0, 0], [4, 5, 6, 7, 0], [3, 0, 0, 0, 0]])
encoder_len = [3, 4, 1]
decoder_input = tf.constant(
    [[1, 1, 0, 0, 0], [2, 3, 0, 0, 0], [3, 4, 5, 6, 0]])
decoder_len = np.array([2, 2, 4]) + 2
decoder_target = tf.constant(
    [[1, 1, 0, 0, 0], [2, 3, 0, 0, 0], [3, 4, 5, 6, 0]])
start_tokens = tf.ones(shape=[3, 1], dtype=tf.int32, name="start_tokens")
end_tokens = tf.ones(shape=[3, 1], dtype=tf.int32, name="end_tokens")
decoder_input = tf.concat([start_tokens, decoder_input], axis=1)
decoder_target = tf.concat([decoder_target, end_tokens], axis=1)
embedding = tf.get_variable(shape=[8, 3], name="e")
embedded_encoder_input = tf.nn.embedding_lookup(embedding, encoder_input)
with tf.variable_scope("encoder"):
    cells_fw = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(3)]
    cells_bw = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(3)]
    cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
    (fw_outputs, bw_outputs), (fw_state, bw_state) = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                        embedded_encoder_input,
                                        sequence_length=encoder_len,
                                        dtype=tf.float32)
    states = tf.concat([fw_state, bw_state], axis=2)
    init_states = []
    for i in range(3):
        init_states.append(states[i])
    outputs = tf.concat([fw_outputs, bw_outputs], axis=2)
with tf.variable_scope("decoder"):
    cells = [tf.nn.rnn_cell.GRUCell(state_size * 2) for _ in range(3)]
    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    init_states = tuple(init_states)
    # outputs, states = tf.nn.dynamic_rnn(decoder_cell, outputs,
    #                                     sequence_length=decoder_len,
    #                                     initial_state=init_states)
    decoder_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
    training_helper = seq2seq.TrainingHelper(decoder_embedded, decoder_len,
                                             time_major=False)
    output_layer = Dense(decoder_vocab_size)

    decoder = seq2seq.BasicDecoder(decoder_cell, initial_state=init_states,
                                   output_layer=output_layer,
                                   helper=training_helper)
    outputs, last_state, _ = \
        seq2seq.dynamic_decode(decoder=decoder,
                               maximum_iterations=max(decoder_len))
    # [batch_size, max_time_steps, decoder_vocab_size]
    # logits = layers.fully_connected(outputs.rnn_output, decoder_vocab_size,
    #                                 activation_fn=None)
    weights = tf.sequence_mask(decoder_len, dtype=tf.float32)
    loss = seq2seq.sequence_loss(logits=outputs.rnn_output,
                                 targets=decoder_target,
                                 average_across_timesteps=True, weights=weights,
                                 average_across_batch=True)

    # init_state = [state for state in states]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # print(sess.run(tf.shape(fw_outputs)))
    # print(sess.run(fw_outputs))
    # print(sess.run(bw_outputs))
    # print('-----------')
    # print(sess.run(tf.shape(fw_state[-1])))
    # print(sess.run(tf.shape(fw_state)))
    # print(sess.run(fw_state))
    print(sess.run(tf.shape(decoder_embedded)))
    print(sess.run(tf.shape(outputs.rnn_output)))
    print(sess.run(loss))
