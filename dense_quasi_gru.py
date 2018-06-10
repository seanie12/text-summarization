import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.contrib import seq2seq
from tensorflow.contrib import layers
from tensorflow.python.util import nest
from tensorflow.python.layers.core import Dense
from attention_decoder import attention_decoder
import data_util
from qrnn import QRNN_layer


class DenseQuasiGRU(object):
    def __init__(self, vocab_size, embedding_size, state_size, num_layers,
                 filter_width, zoneout, decoder_vocab_size,
                 attention_hidden_size, mode, beam_depth,
                 learning_rate, max_iter=100, attention_mode="Bahdanau"):
        # vocab for both encoder and decoder
        self.vocab_size = vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.mode = mode
        self.filter_width = filter_width
        self.zoneout = zoneout
        self.beam_depth = beam_depth
        self.lr = learning_rate
        self.attention_hidden_size = attention_hidden_size
        self.max_iter = max_iter
        assert attention_mode in ["Bahdanau", "Luong"]
        self.attention_mode = attention_mode
        self.global_step = tf.Variable(tf.constant(0),
                                       trainable=False,
                                       name="global_step")

        # embedding matrix for encoder and decoder
        self.embedding_matrix = tf.get_variable(
            shape=[self.vocab_size, self.embedding_size],
            initializer=layers.xavier_initializer(),
            name="embedding_encoder")
        self.build_model()
        self.summary_op = tf.summary.merge_all()

    def build_model(self):
        self.init_placeholder()
        self.build_encoder()
        self.build_decoder()

    def init_placeholder(self):
        # encoder input: [batch_size, max_time_steps],
        # encoder_len : actual encoder length except padding
        self.encoder_input = tf.placeholder(shape=[None, None], dtype=tf.int32,
                                            name="encoder_input")
        self.encoder_len = tf.placeholder(shape=[None], dtype=tf.int32,
                                          name="encoder_len")
        # target : [batch_size, max_time_steps]
        self.target = tf.placeholder(shape=[None, None], dtype=tf.int32,
                                     name="decoder_input")
        self.batch_size = tf.shape(self.encoder_input)[0]

        decoder_start_token = tf.ones(shape=[self.batch_size, 1],
                                      dtype=tf.int32,
                                      name="start_token") * data_util.ID_GO
        decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32,
                                    name="end_token") * data_util.ID_EOS
        # decode input : [batch_size, max_time_steps + 1] starts with Go symbol
        # ex) GO A B C
        self.decoder_input = tf.concat([decoder_start_token, self.target],
                                       axis=1)
        # decoder target : [batch_size, max_time_steps + 1] ends with EOS symbol
        self.decoder_target = tf.concat([self.target, decoder_end_token],
                                        axis=1)
        # decoder len counts the token except Go and EOS symbol
        # so we need to add 1
        self.decoder_len = tf.placeholder(shape=[None], dtype=tf.int32,
                                          name="decoder_len")
        self.decoder_train_len = self.decoder_len + 1
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                name="dropout")

    def build_encoder(self):
        with tf.variable_scope("encoder") and tf.device("/device:GPU:1"):
            cell_fw_list = [tf.nn.rnn_cell.ResidualWrapper(
                tf.nn.rnn_cell.GRUCell(self.state_size)) for _ in
                range(self.num_layers)]
            cell_bw_list = [tf.nn.rnn_cell.ResidualWrapper(
                tf.nn.rnn_cell.GRUCell(self.state_size)) for _ in
                range(self.num_layers)]
            # apply dropout during training (double check)
            # shared word embedding matrix for encoder and decoder
            embedded_encoder_input = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_input)

            curr_input = embedded_encoder_input
            self.encoder_last_states = []
            for i in range(self.num_layers):
                qrnn = QRNN_layer(self.state_size, fwidth=self.filter_width,
                                  sequence_lengths=self.encoder_len,
                                  pool_type="ifo", zoneout=self.zoneout,
                                  infer=self.train == "test",
                                  name="QRNN_{}".format(i), mask=False)
                qrnn_hidden, state = qrnn(curr_input)
                qrnn_hidden = tf.nn.dropout(qrnn_hidden, self.dropout_keep_prob)
                self.encoder_last_states.append(state)
                # dense connection
                if i < self.num_layers - 1:
                    curr_input = tf.concat([curr_input, qrnn_hidden], axis=2)
                else:
                    self.encoder_outputs = tf.concat([curr_input, qrnn_hidden],
                                                     axis=2)
            # concatenate forward and backward output
            # qrnn_hidden : [batch_size, max_time_step, state_size ]
            # encoder_last_states : [num_layers, batch_size, state_size ]

    def build_decoder(self):
        with tf.variable_scope("decoder"):
            decoder_cell, decoder_initial_state = self.build_decoder_cell()

            # start tokens : [batch_size], which is fed to BeamsearchDecoder during inference
            start_tokens = tf.ones([self.batch_size],
                                   dtype=tf.int32) * data_util.ID_GO
            end_token = data_util.ID_EOS
            input_layer = Dense(self.state_size, dtype=tf.float32,
                                name="input_layer")
            output_layer = Dense(self.decoder_vocab_size,
                                 name="output_projection")
            if self.mode == "train":
                # feed ground truth decoder input token every time step
                decoder_input_lookup = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.decoder_input)
                decoder_input_lookup = input_layer(decoder_input_lookup)
                training_helper = seq2seq.TrainingHelper(
                    inputs=decoder_input_lookup,
                    sequence_length=self.decoder_train_len,
                    name="training_helper")
                training_decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                                        initial_state=decoder_initial_state,
                                                        helper=training_helper,
                                                        output_layer=output_layer)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
                #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                max_decoder_len = tf.reduce_max(self.decoder_train_len)
                decoder_outputs_train, final_state, _ = seq2seq.dynamic_decode(
                    training_decoder, impute_finished=True, swap_memory=True,
                    maximum_iterations=max_decoder_len)
                self.decoder_logits_train = tf.identity(
                    decoder_outputs_train.rnn_output)
                decoder_pred = tf.argmax(self.decoder_logits_train, axis=2)
                # sequence mask for get valid sequence except zero padding
                weights = tf.sequence_mask(self.decoder_len,
                                           maxlen=max_decoder_len,
                                           dtype=tf.float32)
                # compute cross entropy loss for all sequence prediction and ignore loss from zero padding
                self.loss = seq2seq.sequence_loss(
                    logits=self.decoder_logits_train,
                    targets=self.decoder_target,
                    weights=weights, average_across_batch=True,
                    average_across_timesteps=True)
                tf.summary.scalar("loss", self.loss)

                with tf.variable_scope("train_optimizer") and tf.device(
                        "/device:GPU:1"):
                    # use AdamOptimizer and clip gradient by max_norm 5.0
                    # use global step for counting every iteration
                    params = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, params)
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                                  5.0)
                    learning_rate = tf.train.exponential_decay(self.lr,
                                                               self.global_step,
                                                               100000, 0.96)
                    opt = tf.train.AdagradOptimizer(learning_rate)

                    self.train_op = opt.apply_gradients(
                        zip(clipped_gradients, params),
                        global_step=self.global_step)

            elif self.mode == "test":
                def embedding_proj(inputs):
                    return input_layer(
                        tf.nn.embedding_lookup(self.embedding_matrix,
                                               inputs))

                inference_decoder = seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                              embedding=embedding_proj,
                                                              start_tokens=start_tokens,
                                                              end_token=end_token,
                                                              initial_state=decoder_initial_state,
                                                              beam_width=self.beam_depth,
                                                              output_layer=output_layer)

                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #                         namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
                #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
                #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True

                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #                         namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
                #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
                # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
                #                                                    namedtuple(scores, predicted_ids, parent_ids)
                with tf.device("/device:GPU:1"):
                    decoder_outputs, decoder_last_state, decoder_output_length = \
                        seq2seq.dynamic_decode(decoder=inference_decoder,
                                               output_time_major=False,
                                               swap_memory=False,
                                               maximum_iterations=self.max_iter)
                    self.decoder_pred_test = decoder_outputs.predicted_ids

    def build_decoder_cell(self):
        encoder_outputs = self.encoder_outputs
        encoder_last_states = self.encoder_last_states
        encoder_len = self.encoder_len
        # for beam search copy the batch by beam depth times
        if self.mode == "test":
            encoder_outputs = seq2seq.tile_batch(encoder_outputs,
                                                 multiplier=self.beam_depth)
            encoder_last_states = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_depth),
                encoder_last_states)
            encoder_len = seq2seq.tile_batch(self.encoder_len,
                                             self.beam_depth)

        # Bahdanau attention
        self.attention_mechanism = seq2seq.BahdanauAttention(
            num_units=self.state_size,
            memory=encoder_outputs,
            memory_sequence_length=encoder_len)
        # Luong attention
        if self.attention_mode == "Luong":
            self.attention_mechanism = seq2seq.LuongAttention(
                num_units=self.state_size,
                memory=encoder_outputs,
                memory_sequence_length=encoder_len)
        # instantiate decoder cells (uni-directional multi GRU cell)
        decoder_cell_list = [tf.nn.rnn_cell.ResidualWrapper(
            tf.nn.rnn_cell.GRUCell(self.state_size)) for _
            in range(self.num_layers)]
        # apply dropout during training
        if self.mode == "train":
            for i in range(self.num_layers):
                decoder_cell_list[i] = DropoutWrapper(decoder_cell_list[i],
                                                      self.dropout_keep_prob)

        # essential for skip connection
        def atten_decoder_input_fn(inputs, attention):
            _input_layer = Dense(self.state_size)
            return _input_layer(tf.concat([inputs, attention], 1))

        # we only apply attention to last layer of encoder
        decoder_cell_list[-1] = seq2seq.AttentionWrapper(decoder_cell_list[-1],
                                                         self.attention_mechanism,
                                                         self.state_size,
                                                         cell_input_fn=atten_decoder_input_fn)

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state
        # if test mode every batch should be copied by beam depth times
        if self.mode == "train":
            batch_size = self.batch_size
        else:
            batch_size = self.batch_size * self.beam_depth
        init_state = []

        for i in range(self.num_layers):
            init_state.append(encoder_last_states[i])
        init_state[-1] = decoder_cell_list[-1].zero_state(batch_size,
                                                          dtype=tf.float32)
        decoder_init_state = tuple(init_state)
        # decoder_init_state = encoder_last_states
        return tf.nn.rnn_cell.MultiRNNCell(
            decoder_cell_list), decoder_init_state

    def save(self, sess, path, global_step):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        save_path = saver.save(sess, path, global_step)
        print("save session to {}".format(save_path))

    def restore(self, sess, path):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, save_path=path)
        print("load model from : {}".format(path))

    def train(self, sess, encoder_inputs, encoder_length, decoder_inputs,
              decoder_length, dropout):
        """
        :param sess:  tensorflow Session
        :param encoder_inputs: [batch_size, max_time_step]
        :param encoder_length: actual length for encoder input
        :param decoder_inputs: [batch_size, max_time_step] for every batch it starts with GO symbol
        :param decoder_targets: [batch_size, max_time_step] for every batch it ends with EOS symbol
        :param decoder_length: actual length for decoder input
        :return:
        """
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(encoder_inputs, encoder_length,
                                      decoder_inputs, decoder_length, False)
        # Input feeds for dropout
        input_feed[self.dropout_keep_prob] = dropout

        output_feed = [self.train_op,  # Update Op that does optimization
                       self.loss,  # Loss for current batch
                       self.summary_op]  # Training summary

        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]  # loss, summary

    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        """Run a evaluation step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length,
                                      False)
        # Input feeds for dropout
        input_feed[self.dropout_keep_prob] = 1.0

        output_feed = [self.loss,  # Loss for current batch
                       self.summary_op]  # Training summary
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]  # loss

    def predict(self, sess, encoder_inputs, encoder_inputs_length):

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs=None,
                                      decoder_inputs_length=None,
                                      decode=True)

        # Input feeds for dropout
        input_feed[self.dropout_keep_prob] = 1.0

        output_feed = [self.decoder_pred_test]
        outputs = sess.run(output_feed, input_feed)
        # GreedyDecoder: [batch_size, max_time_step]
        # BeamSearchDecoder: [batch_size, max_time_step, beam_width]
        return outputs[0]

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """

        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError(
                "Encoder inputs and their lengths must be equal in their "
                "batch_size, %d != %d" % (
                    input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError(
                    "Encoder inputs and Decoder inputs must be equal in their "
                    "batch_size, %d != %d" % (
                        input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError(
                    "Decoder targets and their lengths must be equal in their "
                    "batch_size, %d != %d" % (
                        target_batch_size, decoder_inputs_length.shape[0]))
        input_feed = dict()

        input_feed[self.encoder_input] = encoder_inputs
        input_feed[self.encoder_len] = encoder_inputs_length

        if not decode:
            input_feed[self.target] = decoder_inputs
            input_feed[self.decoder_len] = decoder_inputs_length

        return input_feed


class HybridSummarizationModel(object):
    def __init__(self, vocab_size, embedding_size, state_size, num_layers,
                 mode, beam_depth, learning_rate, max_decode_steps=100):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.attention_hidden_size = state_size
        assert mode in ["train, decode"]
        self.mode = mode
        self.beam_depth = beam_depth
        self.lr = learning_rate
        self.max_decode_steps = max_decode_steps

    def add_placeholder(self):
        self.encoder_input = tf.placeholder(shape=[None, None], dtype=tf.int32,
                                            name="encoder_input")
        self.encoder_len = tf.placeholder(shape=[None], dtype=tf.int32,
                                          name="encoder_len")
        encoder_max_len = tf.reduce_max(self.encoder_len)
        self.encoder_padding_mask = tf.sequence_mask(self.encoder_len,
                                                     encoder_max_len,
                                                     dtype=tf.float32)
        self.decoder_input = tf.placeholder(shape=[None, self.max_decode_steps],
                                            dtype=tf.int32,
                                            name="decoder_input")
        self.target = tf.placeholder(shape=[None, self.max_decode_steps - 1],
                                     dtype=tf.int32,
                                     name="decoder_input")
        self.batch_size = tf.shape(self.encoder_input)[0]
        decoder_start_token = tf.ones(shape=[self.batch_size, 1],
                                      dtype=tf.int32,
                                      name="start_token") * data_util.ID_GO
        decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32,
                                    name="end_token") * data_util.ID_EOS
        self.decoder_input = tf.concat([decoder_start_token, self.target],
                                       axis=1)
        self.decoder_target = tf.concat([self.target, decoder_end_token],
                                        axis=1)
        self.decoder_len = tf.placeholder(shape=[None], dtype=tf.int32,
                                          name="decoder_len")

        decoder_max_len = tf.reduce_max(self.decoder_len)
        self.decoder_padding_mask = tf.sequence_mask(self.decoder_len,
                                                     decoder_max_len)

    def _add_encoder(self, encoder_inputs, seq_len):
        """

        :param encoder_inputs: a tensor shape of
        [batch_size, max_time_step, embedding_size]
        :param seq_len: sequence length of each data
        :return:
            encoder_outputs
        """
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.state_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.state_size)
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, encoder_inputs, dtype=tf.float32,
            sequence_length=seq_len, swap_memory=True)
        # concat forward and backward state
        encoder_outputs = tf.concat(encoder_outputs, axis=2)
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        """

        :param fw_st: lstm tuple forward
        :param bw_st: lstm tupe backward
        :return: reduced lstm tuple with state size from state_size *2
        """
        with tf.variable_scope("reduce_final_st"):
            w_reduce_c = tf.get_variable(
                shape=[self.state_size * 2, self.state_size],
                initializer=layers.xavier_initializer())
            w_reduce_h = tf.get_variable(
                shape=[self.state_size * 2, self.state_size],
                initializer=layers.xavier_initializer(), name="reduce_h")
            bias_c = tf.get_variable(shape=[self.state_size],
                                     initializer=layers.xavier_initializer(),
                                     name="bias_c")
            bias_h = tf.get_variable(shape=[self.state_size],
                                     initializer=layers.xavier_initializer(),
                                     name="bias_h")
            # concatenate forward and backward state
            old_cell_state = tf.concat([fw_st.c, bw_st.c], axis=1)
            old_hidden_state = tf.concat([fw_st.h, bw_st.h], axis=1)

            new_cell = tf.nn.elu(
                tf.matmul(old_cell_state, w_reduce_c) + bias_c)
            new_state = tf.nn.elu(
                tf.matmul(old_hidden_state, w_reduce_h) + bias_h)
            return tf.nn.rnn_cell.LSTMStateTuple(new_cell, new_state)

    def _add_decoder(self, decoder_input):
        """

        :param decoder_input: list of tensor shape [batch, embedding_size]
        :return:
            outputs : list of tensor, output of each time step decoder
            out_state : final state of decoder
            attn_dists : attention distribution for all time step
        """
        # uni-directional LSTM
        cell = tf.nn.rnn_cell.LSTMCell(self.state_size)
        # attention_decoder(decoder_inputs, initial_state, encoder_states,
        #                 encoder_padding_mask, cell, init_state_attention=False)
        init_attention = self.mode == "decode"
        outputs, out_state, attn_dists = attention_decoder(decoder_input,
                                                           self.decoder_init,
                                                           self.encoder_states,
                                                           self.encoder_padding_mask,
                                                           cell, init_attention)

        return outputs, out_state, attn_dists

    def _add_seq2seq(self):
        with tf.variable_scope("seq2seq"):
            embedding = tf.get_variable(
                shape=[self.vocab_size, self.embedding_size],
                initializer=layers.xavier_initializer(), name="embedding")
            # [batch_size, max_time_step, embedding_size]
            encoder_input = tf.nn.embedding_lookup(embedding,
                                                   self.encoder_input)
            # list of [batch_size, embedding_size],
            # length of list is max_decode_step
            decoder_input = [tf.nn.embedding_lookup(embedding, decoder_batch)
                             for decoder_batch in
                             tf.unstack(self.encoder_input, axis=1)]
            with tf.variable_scope("encoder") and tf.device("/device/GPU:0"):
                encoder_outputs, fw_st, bw_st = self._add_encoder(encoder_input,
                                                                  self.encoder_len)
                self.encoder_states = encoder_outputs
                self.decoder_init = self._reduce_states(fw_st, bw_st)

            with tf.variable_scope("decoder") and tf.device("device/GPU:1"):
                decoder_outputs, self.decoder_output_state, \
                self.attention_dists = self._add_decoder(decoder_input)

            with tf.variable_scope("output_projection"):
                w_out = tf.get_variable(
                    shape=[self.state_size, self.vocab_size],
                    initializer=layers.xavier_initializer(), name="w_out")
                b_out = tf.get_variable(shape=[self.vocab_size],
                                        initializer=layers.xavier_initializer(),
                                        name="b_out")
                # un-normalized vocab distribution for all time step
                vocab_scores = []
                for i, decoder_output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    vocab_scores.append(
                        tf.matmul(decoder_output, w_out) + b_out)

                vocab_dists = [tf.nn.softmax(score) for score in vocab_scores]
            if self.mode == "train":
                with tf.variable_scope("loss"):
                    # change vocab scores from [max_time_stpe, batch, vocab_size]
                    # to [batch, max_time_step, vocab_size] using tf.stack
                    # sequence_loss apply softmax internally
                    self.loss = tf.contrib.seq2seq.sequence_loss(
                        tf.stack(vocab_scores, axis=1), self.decoder_target,
                        self.decoder_padding_mask)
                    tf.summary.scalar("loss", self.loss)
            if self.mode == "decode":
                # when decoding, we run beam search every one time step
                assert len(vocab_dists) == 1
                probs = vocab_dists[0]
                topk_probs, self.topk_ids = tf.nn.top_k(probs,
                                                        self.beam_depth * 2)
                self.topk_probs = tf.log(topk_probs)

    def _add_train_op(self):
        vars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, vars)
        clipped_grads, global_norm = tf.clip_by_global_norm(gradients, 5.0)

        optimizer = tf.train.AdagradOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(clipped_grads, vars),
                                                  global_step=self.global_step)

    def build_graph(self):
        self.add_placeholder()
        self._add_seq2seq()
        self.global_step = tf.Variable(tf.constant(0), trainable=False,
                                       name="global_step")
        if self.mode == "train":
            self._add_train_op()
        self.summaries = tf.summary.merge_all()
