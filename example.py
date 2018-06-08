import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

hparams = tf.contrib.training.HParams(
    batch_size=3,
    encoder_length=4,
    decoder_length=5,
    num_units=6,
    src_vocab_size=7,
    embedding_size=8,
    tgt_vocab_size=9,
    learning_rate = 0.01,
    max_gradient_norm = 5.0,
    beam_width =10,
    use_attention = False,
)

# Symbol for start decode process.
tgt_sos_id = 7

# Symbol for end of decode process.
tgt_eos_id = 8

# For debug purpose.
tf.reset_default_graph()

# Encoder
#   encoder_inputs: [encoder_length, batch_size]
#   This is time major where encoder_length comes first instead of batch_size.
encoder_inputs = tf.placeholder(tf.int32, shape=(hparams.encoder_length, hparams.batch_size), name="encoder_inputs")

# Embedding
#   Matrix for embedding: [src_vocab_size, embedding_size]
embedding_encoder = tf.get_variable(
    "embedding_encoder", [hparams.src_vocab_size, hparams.embedding_size])

# Look up embedding:
#   encoder_inputs: [encoder_length, batch_size]
#   encoder_emb_inputs: [encoder_length, batch_size, embedding_size]
encoder_emb_inputs = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)

# LSTM cell.
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units)

# Run Dynamic RNN
#   encoder_outputs: [encoder_length, batch_size, num_units]
#   encoder_state: [batch_size, num_units], this is final state of the cell for each batch.
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inputs, time_major=True, dtype=tf.float32)

# Decoder input
#   decoder_inputs: [decoder_length, batch_size]
#   decoder_lengths: [batch_size]
#   This is grand truth target inputs for training.
decoder_inputs = tf.placeholder(tf.int32, shape=(hparams.decoder_length, hparams.batch_size), name="decoder_inputs")
decoder_lengths = tf.placeholder(tf.int32, shape=(hparams.batch_size), name="decoer_length")

# EmbeddingDecoder:
#    Embedding for decoder.
#    This is used to convert encode training target texts to list of ids.
embedding_decoder = tf.get_variable(
    "embedding_decoder", [hparams.tgt_vocab_size, hparams.embedding_size])


# Look up embedding:
#   decoder_inputs: [decoder_length, batch_size]
#   decoder_emb_inp: [decoder_length, batch_size, embedding_size]
decoder_emb_inputs = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

# https://stackoverflow.com/questions/39573188/output-projection-in-seq2seq-model-tensorflow
# Internally, a neural network operates on dense vectors of some size,
# often 256, 512 or 1024 floats (let's say 512 for here).
# But at the end it needs to predict a word from the vocabulary which is often much larger,
# e.g., 40000 words. Output projection is the final linear layer that converts (projects) from the internal representation to the larger one.
# So, for example, it can consist of a 512 x 40000 parameter matrix and a 40000 parameter for the bias vector.
projection_layer = layers_core.Dense(
            hparams.tgt_vocab_size, use_bias=False)

helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, decoder_lengths, time_major=True)

# Decoder with helper:
#   decoder_emb_inputs: [decoder_length, batch_size, embedding_size]
#   decoder_length: [batch_size] vector, which represents each target sequence length.
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units)

if hparams.use_attention:
  # Attention
  # attention_states: [batch_size, max_time, num_units]
  attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

  # Create an attention mechanism
  attention_mechanism = tf.contrib.seq2seq.LuongAttention(
      hparams.num_units, attention_states,
      memory_sequence_length=None)

  decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
      decoder_cell, attention_mechanism,
      attention_layer_size=hparams.num_units)

  initial_state = decoder_cell.zero_state(hparams.batch_size, tf.float32).clone(cell_state=encoder_state)
else:
  initial_state = encoder_state

# Decoder and decode
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, initial_state,
    output_layer=projection_layer)

# Dynamic decoding
#   final_outputs.rnn_output: [batch_size, decoder_length, tgt_vocab_size], list of RNN state.
#   final_outputs.sample_id: [batch_size, decoder_length], list of argmax of rnn_output.
#   final_state: [batch_size, num_units], list of final state of RNN on decode process.
#   final_sequence_lengths: [batch_size], list of each decoded sequence.

final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)

print("rnn_output.shape=", final_outputs.rnn_output.shape)
print("sample_id.shape=", final_outputs.sample_id.shape)
print("final_state=", _final_state)
print("final_sequence_lengths.shape=", _final_sequence_lengths.shape)

logits = final_outputs.rnn_output

# Target labels
#   As described in doc for sparse_softmax_cross_entropy_with_logits,
#   labels should be [batch_size, decoder_lengths] instead of [batch_size, decoder_lengths, tgt_vocab_size].
#   So labels should have indices instead of tgt_vocab_size classes.
target_labels = tf.placeholder(tf.int32, shape=(hparams.batch_size, hparams.decoder_length))

# Loss
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=target_labels, logits=logits)

# Train
global_step = tf.Variable(0, name='global_step', trainable=False)

# Calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, hparams.max_gradient_norm)

# Optimization
optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
train_op = optimizer.apply_gradients(
    zip(clipped_gradients, params), global_step=global_step)

#optimizer = tf.train.GradientDescentOptimizer(hparams.learning_rate)
#train_op = optimizer.minimize(loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training data.

# Tweet
tweet1 = np.array([1, 2, 3, 4])
tweet2 = np.array([0, 5, 6, 3])

# Make batch data.
train_encoder_inputs = np.empty((hparams.encoder_length, hparams.batch_size))
train_encoder_inputs[:, 0] = tweet1
train_encoder_inputs[:, 1] = tweet2
train_encoder_inputs[:, 2] = tweet1
print("Tweets")
print(train_encoder_inputs)

# Reply
training_decoder_input1 = [tgt_sos_id, 2, 3, 4, 5]
training_decoder_input2 = [tgt_sos_id, 5, 6, 4, 3]

training_target_label1 = [2, 3, 4, 5, tgt_eos_id]
training_target_label2 = [5, 6, 4, 3, tgt_eos_id]

training_target_labels = np.empty((hparams.batch_size, hparams.decoder_length))
training_target_labels[0] = training_target_label1
training_target_labels[1] = training_target_label2
training_target_labels[2] = training_target_label1
print("Replies")
print(training_target_labels)

training_decoder_inputs = np.empty((hparams.decoder_length, hparams.batch_size))
training_decoder_inputs[:, 0] = training_decoder_input1
training_decoder_inputs[:, 1] = training_decoder_input2
training_decoder_inputs[:, 2] = training_decoder_input1
print(training_decoder_inputs)

feed_dict = {
    encoder_inputs: train_encoder_inputs,
    target_labels: training_target_labels,
    decoder_inputs: training_decoder_inputs,
    decoder_lengths: np.ones((hparams.batch_size), dtype=int) * hparams.decoder_length
}

# Train
for i in range(100):
  _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

# Inference
inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding_decoder,
    tf.fill([hparams.batch_size], tgt_sos_id), tgt_eos_id)

# Inference Decoder
inference_decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, inference_helper, initial_state,
    output_layer=projection_layer)


# We should specify maximum_iterations, it can't stop otherwise.
source_sequence_length = hparams.encoder_length
maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)

# Dynamic decoding
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    inference_decoder, maximum_iterations=maximum_iterations)
translations = outputs.sample_id

# Input tweets
inference_encoder_inputs = np.empty((hparams.encoder_length, hparams.batch_size))
inference_encoder_inputs[:, 0] = tweet1
inference_encoder_inputs[:, 1] = tweet2
inference_encoder_inputs[:, 2] = tweet1

feed_dict = {
    encoder_inputs: inference_encoder_inputs,
}

replies = sess.run([translations], feed_dict=feed_dict)
print(replies)

# Beam Search
# Replicate encoder infos beam_width times
decoder_initial_state = tf.contrib.seq2seq.tile_batch(
    initial_state, multiplier=hparams.beam_width)

# Define a beam-search decoder
inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding_decoder,
        start_tokens=tf.fill([hparams.batch_size], tgt_sos_id),
        end_token=tgt_eos_id,
        initial_state=decoder_initial_state,
        beam_width=hparams.beam_width,
        output_layer=projection_layer,
        length_penalty_weight=0.0)

# Dynamic decoding
decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
    inference_decoder, maximum_iterations=maximum_iterations)
translations = decoder_outputs.predicted_ids
output_score = decoder_outputs.beam_search_decoder_output.scores
shape = tf.shape(output_score)


replies, decoder_out, decoder_shape = sess.run([translations, output_score, shape], feed_dict=feed_dict)
print("-------------")

print(decoder_out)
print(replies)
print(decoder_shape)