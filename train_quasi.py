import tensorflow as tf
from dense_quasi_gru import DenseQuasiGRU
from data_util import batch_loader, load_data, load_valid_data, \
    make_array_format
import os
import numpy as np

# configuration
nepoch_no_improv = 5
debug = False
max_vocab_size = 5e4
max_num_tokens = 400
learning_rate = 0.15
batch_size = 16
num_epochs = 40
dropout = 0.5
zoneout = 0.1
filter_width = 3
embedding_size = 300
num_layers = 3
summary_len = 100
beam_depth = 4
state_size = 50
mode = "train"
doc_file = "data/modified_train_article.txt"
sum_file = "data/modified_train_abstract.txt"
vocab_file = "data/vocab"
checkpoint_dir = "./save/quasi/checkpoints"
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
def load_glove(glove_file, vocab, embedding_size):
    print("load pretrained glove from : {}".format(glove_file))
    f = open(glove_file, "r", encoding="utf-8")
    lines = f.readlines()
    embedding = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_size))
    for line in lines:
        tokens = line.strip().split()
        word = tokens[0]
        try:
            vector = np.asarray(tokens[1:embedding_size], dtype=np.float32)
            index = vocab.word2idx(word)
            # if unknown word, skip
            if index == 2:
                continue
            embedding[index] = vector
        except:
            continue
    # for PAD token, assign zero vector
    f.close()
    embedding[0] = [0.0] * embedding_size
    return embedding


with tf.Graph().as_default():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    log_writer = tf.summary.FileWriter(checkpoint_dir, graph=sess.graph)
    model = DenseQuasiGRU(vocab_size=vocab_size, embedding_size=embedding_size,
                          num_layers=num_layers, state_size=state_size,
                          filter_width=filter_width, zoneout=zoneout,
                          decoder_vocab_size=vocab_size,
                          attention_hidden_size=state_size, mode=mode,
                          beam_depth=beam_depth, learning_rate=learning_rate,
                          max_iter=summary_len)
    pretrained_embedding = load_glove("data/glove.840B.300d.txt", vocab,
                                      embedding_size)
    model.embedding_matrix.assign(pretrained_embedding)
    del pretrained_embedding
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
            loss, summary_op = model.eval(sess, encoder_input, encoder_len,
                                          decoder_input, decoder_len)
            losses.append(loss)
        result = np.mean(losses)
        if result < best_loss:
            best_loss = result
            print("new record loss : {}".format(result))
            no_improv = 0
            model.save(sess, checkpoint_prefix, step)
        no_improv += 1
        if no_improv == nepoch_no_improv:
            print("no improvement for {} epochs".format(no_improv))
            break

    print("end of training")
