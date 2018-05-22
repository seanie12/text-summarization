from dense_bigru import DenseBiGRU
import data_util
import tensorflow as tf
import os

max_num_tokens = 900
max_vocab_size = 500000
learning_rate = 0.1
batch_size = 32
dropout = 0.5
embedding_size = 300
num_layers = 1
summary_len = 100
attention_hidden_size = 100
beam_depth = 5
state_size = 120
mode = "test"
doc_file = "data/test_article.txt"
sum_file = "data/test_abstract.txt"
vocab_file = "data/vocab"
checkpoint_dir = "./save/baseline/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "baseline")
vocab = data_util.Vocab("data/vocab", max_vocab_size)
docs = data_util.load_test_data(doc_file, vocab, max_num_tokens)
summary_file = "result/summaries.txt"
with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session()
    log_writer = tf.summary.FileWriter(checkpoint_dir, graph=sess.graph)
    model = DenseBiGRU(vocab_size=max_vocab_size, embedding_size=embedding_size,
                       num_layers=num_layers, state_size=state_size,
                       decoder_vocab_size=max_vocab_size,
                       attention_hidden_size=attention_hidden_size, mode=mode,
                       beam_depth=beam_depth, learning_rate=learning_rate,
                       max_iter=summary_len)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reload model parameters")
        model.restore(sess, ckpt.model_checkpoint_path)
    else:
        exit()
    batches = data_util.batch_loader(docs, batch_size)
    fw = open(summary_file, "w", encoding="utf-8")
    for idx, batch in enumerate(batches):
        batch_encoder_input, batch_encoder_len = data_util.make_array_format(
            batch)
        # if BeamSearchDecoder : [batch, max_time_step, beam_Depth]
        batch_summaries = model.predict(sess, batch_encoder_input,
                                        batch_encoder_len)
        # iterate every example, get best beam search result
        for summary in batch_summaries:
            best_summary = summary[:, 0]
            # convert indices to corresponding tokens
            summary_tokens = data_util.map_idx2tok(best_summary, vocab)
            # remove EOS token
            try:
                stop_index = summary_tokens.index(data_util.ID_EOS)
                summary_tokens = summary_tokens[:stop_index]
            except ValueError:
                summary_tokens = summary_tokens
            results = " ".join(summary_tokens)
            fw.write(results + "\n")
        print("decoding {}th batch from data".format(idx))
    print("end of decoding")
    fw.close()
