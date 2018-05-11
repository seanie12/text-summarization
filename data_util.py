from collections import Counter
import itertools
import pickle
import numpy as np

# EOS and PAD are merged into EOS
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_GO, MARK_EOS, MARK_UNK]
ID_EOS = 0
ID_GO = 1
ID_UNK = 2

base_tok2idx = {
    MARK_GO: ID_GO,
    MARK_EOS: ID_EOS,
    MARK_UNK: ID_UNK
}
base_idx2tok = {
    ID_GO: MARK_GO,
    ID_EOS: MARK_EOS,
    ID_UNK: MARK_UNK
}


def load_dict(file_path):
    print("load dictionary from :{}".format(file_path))
    try:
        f = open(file_path, "rb")
        tok2idx, idx2tok, decoder_vocab_size = pickle.load(f)
        return tok2idx, idx2tok, decoder_vocab_size
    except FileNotFoundError:
        return None


def create_dict(docs, offset, max_words=None):
    """
    :param docs: list of tokenized list
    :param offset: starting index
    :param max_words: limit of vocab size
    :return: a tuple of dictionaries (tok2idx, idx2tok)
    """
    # flatten 2d list into 1d for word counting
    docs = list(itertools.chain.from_iterable(docs))
    counter = Counter(docs)
    if max_words:
        words = counter.most_common(max_words)
    else:
        words = counter.most_common()
    words, freq = zip(*words)
    # words = MARKS + list(words)
    tok2idx = {word.strip(): offset + i for i, word in enumerate(words)}
    idx2tok = {idx: tok for (tok, idx) in tok2idx.items()}
    return tok2idx, idx2tok


def load_data(doc_file, sum_file, dict_file,
              max_doc_vocab=None, max_sum_vocab=None, debug=False):
    print("load data")
    with open(doc_file, "r", encoding="utf-8") as doc_file:
        docs = doc_file.readlines()
    with open(sum_file, "r", encoding="utf-8") as sum_file:
        sums = sum_file.readlines()
    if debug:
        docs = docs[:500]
        sums = sums[:500]
    # check whether the number of documents and summaries are the same
    assert len(docs) == len(sums)
    # split document into word level tokens
    docs = list(map(lambda doc: doc.split(), docs))
    sums = list(map(lambda summary: summary.split(), sums))
    # load saved dictionary
    dicts = load_dict(dict_file)

    # create dictionary which map unique token to index
    if dicts is None:
        tok2idx, idx2tok = create_dict(sums, offset=len(base_tok2idx))
        tok2idx.update(base_tok2idx)
        idx2tok.update(base_idx2tok)
        decoder_vocab_size = len(tok2idx)
        dicts = (tok2idx, idx2tok)
        # append (token, idx) entries which does not appear in source document
        tok2idx, idx2tok = update_dict(docs, dicts, offset=len(tok2idx))
        # save the dictionary as pickle
        with open(dict_file, "wb") as f:
            dicts = (tok2idx, idx2tok, decoder_vocab_size)
            pickle.dump(dicts, f)
    else:
        tok2idx, idx2tok, decoder_vocab_size = dicts
    vectorized_docs = map_corpus2idx(docs, tok2idx)
    vectorized_sums = map_corpus2idx(sums, tok2idx)
    return vectorized_docs, vectorized_sums, tok2idx, idx2tok, decoder_vocab_size


def update_dict(sums, dicts, offset):
    idx = offset
    tok2idx, idx2tok = dicts
    for summary in sums:
        for word in summary:
            if word not in tok2idx:
                tok2idx[word] = idx
                idx2tok[idx] = word
                idx += 1
    return tok2idx, idx2tok


def load_valid_data(doc_file: str, sum_file: str, tok2idx: dict):
    with open(doc_file, "r", encoding="utf-8") as doc_file:
        docs = doc_file.readlines()
    with open(sum_file, 'r', encoding="utf-8") as sum_file:
        summaries = sum_file.readlines()
    docs = list(map(lambda doc: doc.split(), docs))
    summaries = list(map(lambda summary: summary.split(), summaries))
    vectorized_docs = map_corpus2idx(docs, tok2idx)
    vectorized_summaries = map_corpus2idx(summaries, tok2idx)
    return vectorized_docs, vectorized_summaries


def load_test_data(doc_file: str, tok2idx):
    with open(doc_file, "r", encoding="utf-8") as doc_file:
        docs = doc_file.readlines()
    vectorized_docs = map_corpus2idx(docs, tok2idx)
    return vectorized_docs


def map_idx2tok(sentence: str, idx2tok: dict):
    # convert indices in one sentence to corresponding token
    return list(map(lambda x: idx2tok[x], sentence))


def map_corpus2idx(corpus: list, tok2idx: dict):
    """
    :param corpus: list of document which is composed of tokens
    :param tok2idx: dictionary which map token to index
    :return: list of indices
    """
    vectorized_corpus = []
    # sort document by the number of tokens for each document
    corpus = sorted(corpus, key=lambda x: len(x))
    for doc in corpus:
        buffer_doc = []
        for word in doc:
            if word in tok2idx:
                buffer_doc.append(tok2idx[word.strip()])
            else:
                buffer_doc.append(tok2idx[ID_UNK])
        vectorized_corpus.append(buffer_doc)
    return vectorized_corpus


def batch_loader(iterable, batch_size):
    length = len(iterable)
    # need shuffle
    for start_idx in range(0, length, batch_size):
        yield iterable[start_idx: min(start_idx + batch_size, length)]


def _make_array_format(source, target=None):
    """

    :param source: articles to be summarized
    :param target: summaries of article
    :return: source,  target (all zero padded)
    """
    encoder_len = np.array([len(article) for article in source])
    encoder_input = zero_pad(source, max(encoder_len))

    if target:
        assert len(source) == len(target)
        decoder_len = np.array([len(summary) for summary in target])
        target = zero_pad(target, max(decoder_len))
        return encoder_input, encoder_len, target, decoder_len
    else:
        return encoder_input, encoder_len


def zero_pad(docs, max_len):
    padded_docs = list(
        map(lambda doc: doc + [ID_EOS] * (max_len - len(doc)), docs))
    return np.array(padded_docs)


if __name__ == "__main__":
    vectorized_docs, vectorized_sums, tok2idx, idx2tok, decoder_vocab_size = load_data(
        "article.txt", "summary.txt",
        "vocab.pickle")
    print(vectorized_docs)
    print(vectorized_sums)
    print(tok2idx)
    print(idx2tok)
