import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.contrib import seq2seq
from tensorflow.contrib import layers
from tensorflow.python.util import nest
from tensorflow.python.layers.core import Dense
import data_util


class DenseBiGRU(object):
    def __init__(self, vocab_size, embedding_size, state_size, num_layers,
                 decoder_vocab_size, attention_hidden_size, mode, beam_depth,
                 learning_rate, max_iter=100, attention_mode="Bahdanau"):
        # vocab for both encoder and decoder
        self.vocab_size = vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.num_layers = num_layers
        self.mode = mode
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
        with tf.variable_scope("encoder"):
            cell_fw_list = [tf.nn.rnn_cell.GRUCell(self.state_size) for _ in
                            range(self.num_layers)]
            cell_bw_list = [tf.nn.rnn_cell.GRUCell(self.state_size) for _ in
                            range(self.num_layers)]
            # apply dropout during training (double check)
            if self.mode == "train":
                for i in range(self.num_layers):
                    cell_fw_list[i] = DropoutWrapper(cell_fw_list[i],
                                                     self.dropout_keep_prob)
                    cell_bw_list[i] = DropoutWrapper(cell_bw_list[i],
                                                     self.dropout_keep_prob)
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw_list)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw_list)
            # shared word embedding matrix for encoder and decoder

            embedded_encoder_input = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_input)
            ((fw_output, bw_output),
             (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                     cell_bw,
                                                                     inputs=embedded_encoder_input,
                                                                     dtype=tf.float32,
                                                                     sequence_length=self.encoder_len
                                                                     )
            # concatenate forward and backward output
            # *_output : [batch_size, max_time_step, state_size ] -> [batch, time_step, state_size *2]
            # *_state : [num_layers, batch_size, state_size ] -> [batch, time_step, state_size * 2]
            self.encoder_outputs = tf.concat([fw_output, bw_output], axis=2)
            self.encoder_last_states = tf.concat([fw_state, bw_state], axis=2)

    def build_decoder(self):
        with tf.variable_scope("decoder"):
            decoder_cell, decoder_initial_state = self.build_decoder_cell()

            # start tokens : [batch_size], which is fed to BeamsearchDecoder during inference
            start_tokens = tf.ones([self.batch_size],
                                   dtype=tf.int32) * data_util.ID_GO
            end_token = data_util.ID_EOS
            output_layer = Dense(self.decoder_vocab_size,
                                 name="output_projection")
            if self.mode == "train":
                # feed ground truth decoder input token every time step
                decoder_input_lookup = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.decoder_input)
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
                    training_decoder, impute_finished=True,
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

                with tf.variable_scope("train_optimizer"):
                    # use AdamOptimizer and clip gradient by max_norm 5.0
                    # use global step for counting every iteration
                    params = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, params)
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                                  5.0)
                    opt = tf.train.AdadeltaOptimizer(self.lr)
                    self.train_op = opt.apply_gradients(
                        zip(clipped_gradients, params),
                        global_step=self.global_step)

            elif self.mode == "test":
                def embedding_proj(inputs):
                    return tf.nn.embedding_lookup(self.embedding_matrix,
                                                  inputs)

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
                decoder_outputs, decoder_last_state, decoder_outputs_length = \
                    seq2seq.dynamic_decode(decoder=inference_decoder,
                                           output_time_major=False,
                                           maximum_iterations=self.max_iter)
                self.decoder_pred_test = decoder_outputs.predicted_ids

    def build_decoder_cell(self):
        encoder_outputs = self.encoder_outputs
        encoder_last_states = self.encoder_last_states
        # for beam search copy the batch by beam depth times
        if self.mode == "test":
            encoder_outputs = seq2seq.tile_batch(encoder_outputs,
                                                 multiplier=self.beam_depth)
            encoder_last_states = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_depth))
            encoder_len = seq2seq.tile_batch(self.encoder_len,
                                             self.beam_depth)

        # Bahdanau attention
        self.attention_mechanism = seq2seq.BahdanauAttention(
            num_units=self.attention_hidden_size,
            memory=encoder_outputs,
            memory_sequence_length=self.encoder_len)
        # Luong attention
        if self.attention_mode == "Luong":
            self.attention_mechanism = seq2seq.LuongAttention(
                num_units=self.attention_hidden_size,
                memory=encoder_outputs,
                memory_sequence_length=encoder_len)
        # instantiate decoder cells (uni-directional multi GRU cell)
        decoder_cell_list = [tf.nn.rnn_cell.GRUCell(self.state_size * 2) for _
                             in range(self.num_layers)]
        # apply dropout during training
        if self.mode == "train":
            for i in range(self.num_layers):
                decoder_cell_list[i] = DropoutWrapper(decoder_cell_list[i],
                                                      self.dropout_keep_prob)

        # we only apply attention to last layer of encoder
        decoder_cell_list[-1] = seq2seq.AttentionWrapper(decoder_cell_list[-1],
                                                         self.attention_mechanism,
                                                         self.attention_hidden_size)

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
        saver = tf.train.Saver(tf.global_variables())
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
        input_feed = {}

        input_feed[self.encoder_input] = encoder_inputs
        input_feed[self.encoder_len] = encoder_inputs_length

        if not decode:
            input_feed[self.target] = decoder_inputs
            input_feed[self.decoder_len] = decoder_inputs_length

        return input_feed
