import tensorflow as tf
from dense_bigru import DenseBiGRU
from data_util import batch_loader, load_data, load_valid_data, \
    make_array_format
import os
import numpy as np

max_vocab_size = 15e4
learning_rate = 0.1
batch_size = 32
num_epochs = 200
dropout = 0.5
embedding_size = 300
num_layers = 1
summary_len = 100
beam_depth = 5
state_size = 120
mode = "train"
doc_file = "data/train_article.txt"
sum_file = "data/train_abstract.txt"
vocab_file = "data/vocab"
checkpoint_dir = "./save/baseline/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "baseline")
dev_doc_file = "data/val_article.txt"
dev_sum_file = "data/val_abstract.txt"
# load source and target data
docs, sums, vocab = load_data(doc_file, sum_file, vocab_file, max_vocab_size,
                              debug=True)
dev_docs, dev_sums = load_valid_data(dev_doc_file, dev_sum_file, vocab)
vocab_size = vocab.size()
# self, vocab_size, embedding_size, state_size, num_layers,
#                  decoder_vocab_size, attention_hidden_size, mode, beam_depth,
#                  learning_rate, max_iter=100, attention_mode="Bahdanau"):
# TODO : load pretrained vector(GLOVE or word2vec), learning rate decay
with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session()
    log_writer = tf.summary.FileWriter(checkpoint_dir, graph=sess.graph)
    model = DenseBiGRU(vocab_size=vocab_size, embedding_size=embedding_size,
                       num_layers=num_layers, state_size=state_size,
                       decoder_vocab_size=vocab_size,
                       attention_hidden_size=state_size, mode=mode,
                       beam_depth=beam_depth, learning_rate=learning_rate,
                       max_iter=summary_len)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reload model paramters")
        model.restore(sess, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("create new model parameter")
        sess.run(tf.global_variables_initializer())
    step = None
    best_result = 1e10
    for epoch in range(num_epochs):
        batches = batch_loader(list(zip(docs, sums)), batch_size)
        dev_batches = batch_loader(
            list(zip(dev_doc_file, dev_sum_file)), batch_size)
        for batch in batches:
            batch_source, batch_target = zip(*batch)
            batch_encoder_input, batch_encoder_len, \
            batch_decoder_input, batch_decoder_len = \
                make_array_format(batch_source, batch_target)
            batch_loss, summary_op = model.train(sess,
                                              encoder_inputs=batch_encoder_input,
                                              encoder_length=batch_encoder_len,
                                              decoder_inputs=batch_decoder_input,
                                              decoder_length=batch_decoder_len,
                                              dropout=dropout)
            step = model.global_step.eval(sess)
            print("epoch : {} step : {}, loss : {}".format(epoch + 1, step,
                                                           batch_loss))
        losses = []
        for batch in dev_batches:
            source, target = zip(*batch)
            encoder_input, encoder_len, decoder_input, \
            decoder_len = make_array_format(source, target)
            model.mode = "test"
            loss, summary_op = model.eval(sess, encoder_input, encoder_len,
                                   decoder_input, decoder_len)
            losses.append(loss)
        result = np.mean(losses)
        if result < best_result:
            best_result = result
            print("new record loss : {}".format(result))
            model.save(sess, checkpoint_prefix, step)
        model.mode = "train"

    print("end of training")
