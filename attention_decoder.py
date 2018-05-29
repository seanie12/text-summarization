import tensorflow as tf
from tensorflow.contrib import layers


def attention_decoder(decoder_inputs: list, initial_state: tf.Tensor,
                      encoder_states, encoder_padding_mask, cell,
                      init_state_attention=False):
    # args
    # decoder inputs : list of tensor, shape : [batch, embeding]
    # initial_state : decoder initial state
    # encoder_states : encoder hidden states
    # encoder_padding_mask padding token is marked as 1 otherwise 0
    # cell : tf.nn.rnn_cell instance
    # i
    with tf.variable_scope("attention_decoder"):
        batch_size, nsteps, state_size = tf.unstack(tf.shape(encoder_states))

        # (batch_size, time_step, 1, state_size )
        encoder_states = tf.expand_dims(encoder_states, axis=2)
        # badhnau attention : v^t *tanh(w_h h_i + w_s * s_t + b)
        attention_size = state_size
        w_h = tf.get_variable(shape=[1, 1, state_size, attention_size],
                              name="w_h",
                              initializer=layers.xavier_initializer())
        encoder_features = tf.nn.conv2d(encoder_states, w_h,
                                        strides=[1, 1, 1, 1], padding="SAME")
        v = tf.get_variable(shape=[attention_size],
                            initializer=layers.xavier_initializer(), name="v")

        def attention(decoder_state):
            with tf.variable_scope("attention"):
                # shape : [batch_size, attention_size]
                # W_s * s_t + b
                decoder_features = linear_layer(decoder_state, attention_size,
                                                "decoder_mat")
                # shape : [batch_size, 1, 1, attention_size]
                decoder_features = tf.expand_dims(
                    tf.expand_dims(decoder_features, 1), 1)

            # apply softmax and mask padding
            def masked_attention(e):
                # args e: un-normalized attention dist
                # shape : [batch_size, attention_size]
                attention_dist = tf.nn.softmax(e)
                # apply mask to zero padding
                attention_dist *= encoder_padding_mask
                masked_sums = tf.reduce_sum(attention_dist, axis=1,
                                            keep_dims=True)
                return attention_dist / masked_sums

            # encoder features : [batch, max_steps, 1, attention_size]
            # decoder features : [batch, 1, 1, attention_size]
            # when two terms are added, broadcast is applied to decoder features
            e = tf.reduce_sum(v * tf.tanh(encoder_features + decoder_features),
                              axis=[2, 3])
            # attention_dist : [batch, max_steps]
            attention_dist = masked_attention(e)
            attention_dist = tf.reshape(attention_dist, [batch_size, -1, 1, 1])
            context_vector = tf.reduce_sum(attention_dist * encoder_states,
                                           axis=[1, 2])
            context_vector = tf.reshape(context_vector, [-1, state_size])
            return context_vector, attention_dist

        outputs = []
        attention_dists = []
        state = initial_state
        context_vector = tf.zeros([batch_size, attention_size])

        if init_state_attention:
            context_vector = attention(state)
        # iterate for every time step of decoder hidden state
        for i, decoder_input in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # concat decoder input and context vector
            decoder_new_input = linear_layer(
                tf.concat([decoder_input, context_vector], axis=1), state_size,
                scope="projected_context")
            # run the decoder rnn cell(instance of tf.nn.rnn_cell)
            cell_output, state = cell(decoder_new_input, state)
            if i == 0 and init_state_attention:
                tf.get_variable_scope().reuse_variables()
                context_vector, attention_dist = attention(state)
            else:
                context_vector, attention_dist = attention(state)
            attention_dists.append(attention_dist)
            # concat context vector and decoder hidden state
            # and multiply V
            with tf.variable_scope("attention_projection") as scope:
                output = linear_layer(
                    tf.concat([cell_output, context_vector], axis=1),
                    state_size, scope)
            outputs.append(output)

            return outputs, state, attention_dist


def linear_layer(input, output_size, scope):
    with tf.variable_scope(scope):
        projected = layers.fully_connected(input, output_size,
                                           activation_fn=None)
        return projected
