import numpy as np

# EOS and PAD are merged into EOS
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_GO, MARK_EOS, MARK_UNK]
ID_EOS = 0
ID_GO = 1
ID_UNK = 2


class Vocab(object):
    # read vocab file and make word2idx and idx2word
    def __init__(self, vocab_file, max_size):
        self._word2idx = dict()
        self._idx2word = dict()
        self._count = 0

        for word in MARKS:
            self._word2idx[word] = self._count
            self._idx2word[self._count] = word
            self._count += 1

        with open(vocab_file, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                tokens = line.split()
                if len(tokens) != 2:
                    continue
                word = tokens[0]
                freq = tokens[1]
                if word in MARKS:
                    raise Exception
                if word in self._word2idx:
                    raise Exception
                self._word2idx[word] = self._count
                self._idx2word[self._count] = word
                self._count += 1
                if self._count >= max_size:
                    break

    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return self._word2idx[MARK_UNK]

    def idx2word(self, word_idx):
        if word_idx not in self._idx2word:
            raise Exception
        return self._idx2word[word_idx]

    def size(self):
        return self._count


def load_data(doc_file, sum_file, vocab_file, max_vocab_size, max_num_tokens,
              debug=False):
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
    docs = list(map(lambda doc: doc.split()[:max_num_tokens], docs))
    # remove <s> and </s> in summary
    sums = list(
        map(lambda summary:
            summary.replace("<s>", "").replace("</s>", "").strip(), sums))
    sums = list(map(lambda summary: summary.split(), sums))
    # load saved dictionary
    vocab = Vocab(vocab_file, max_vocab_size)

    vectorized_docs = map_corpus2idx(docs, vocab)
    vectorized_sums = map_corpus2idx(sums, vocab)
    return vectorized_docs, vectorized_sums, vocab


def load_valid_data(doc_file: str, sum_file: str, vocab: Vocab,
                    max_num_tokens: int):
    # vocab : Vocab object
    with open(doc_file, "r", encoding="utf-8") as doc_file:
        docs = doc_file.readlines()
    with open(sum_file, 'r', encoding="utf-8") as sum_file:
        summaries = sum_file.readlines()
    docs = list(map(lambda doc: doc.split()[:max_num_tokens], docs))
    summaries = list(
        map(lambda summary: summary.split(), summaries))
    vectorized_docs = map_corpus2idx(docs, vocab)
    vectorized_summaries = map_corpus2idx(summaries, vocab)
    return vectorized_docs, vectorized_summaries


def load_test_data(doc_file: str, vocab: Vocab, max_num_tokens: int):
    # vocab : Vocab object
    with open(doc_file, "r", encoding="utf-8") as doc_file:
        docs = doc_file.readlines()
        docs = list(map(lambda doc: doc.split()[:max_num_tokens], docs))
    vectorized_docs = map_corpus2idx(docs, vocab)
    return vectorized_docs


def map_idx2tok(sentence: list, vocab: Vocab):
    # sentence : list of token indices
    # convert indices in one sentence to corresponding token
    return list(map(lambda x: vocab.idx2word(x), sentence))


def map_corpus2idx(corpus: list, vocab: Vocab):
    """
    :param corpus: list of document which is composed of tokens
    :param vocab: Vocab object
    :return: list of indices
    """
    vectorized_corpus = []
    # sort document by the number of tokens for each document
    corpus = sorted(corpus, key=lambda x: len(x))
    for doc in corpus:
        buffer_doc = []
        for word in doc:
            buffer_doc.append(vocab.word2idx(word.strip()))
        vectorized_corpus.append(buffer_doc)
    return vectorized_corpus


def batch_loader(iterable, batch_size):
    length = len(iterable)
    # need shuffle
    for start_idx in range(0, length, batch_size):
        yield iterable[start_idx: min(start_idx + batch_size, length)]


def make_array_format(source, target=None):
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


def remove_empty(doc_file, sum_file, doc_output, sum_output):
    docs = open(doc_file, "r", encoding="utf-8").readlines()
    sums = open(sum_file, "r", encoding="utf-8").readlines()
    modified_doc = open(doc_output, "w", encoding="utf-8")
    modified_sum = open(sum_output, "w", encoding="utf-8")

    for i, (doc, abstract) in enumerate(zip(docs, sums)):
        if doc.strip() == "" or abstract.strip() == "":
            print("empty string at line {}".format(i))
            continue
        modified_doc.write(doc)
        modified_sum.write(abstract)

    modified_doc.close()
    modified_sum.close()


if __name__ == "__main__":
    vocab = Vocab("data/vocab", 1e10)
