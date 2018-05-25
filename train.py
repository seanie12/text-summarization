import tensorflow as tf
from dense_bigru import DenseBiGRU
from data_util import batch_loader, load_data, load_valid_data, \
    make_array_format
import os
import numpy as np

# configuration
nepoch_no_improv = 5
debug = False
max_vocab_size = 5e4
max_num_tokens = 800
learning_rate = 0.001
batch_size = 16
num_epochs = 40
dropout = 0.5
embedding_size = 300
num_layers = 3
summary_len = 100
beam_depth = 4
state_size = 100
mode = "train"
doc_file = "data/modified_train_article.txt"
sum_file = "data/modified_train_abstract.txt"
vocab_file = "data/vocab"
checkpoint_dir = "./save/baseline/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "baseline")
dev_doc_file = "data/val_article.txt"
dev_sum_file = "data/val_abstract.txt"
# load source and target data
docs, sums, vocab = load_data(doc_file, sum_file, vocab_file, max_vocab_size,
                              debug=debug, max_num_tokens=max_num_tokens)
dev_docs, dev_sums = load_valid_data(dev_doc_file, dev_sum_file, vocab,
                                     max_num_tokens)
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
    best_loss = 1e10
    no_improv = 0
    for epoch in range(num_epochs):
        batches = batch_loader(list(zip(docs, sums)), batch_size)
        dev_batches = batch_loader(
            list(zip(dev_docs, dev_sums)), batch_size)
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
        del batches
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
        if result < best_loss:
            best_loss = result
            print("new record loss : {}".format(result))
            no_improv = 0
            model.save(sess, checkpoint_prefix, step)
        model.mode = "train"
        no_improv += 1
        if no_improv == nepoch_no_improv:
            print("no improvement for {} epochs".format(no_improv))
            break

    print("end of training")
