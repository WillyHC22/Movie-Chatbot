import re
import csv
import pickle
import codecs
import logging
import unicodedata

from numpy import True_

logger = logging.getLogger(__name__)

PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_SENTENCE_LENGTH = 10
MIN_COUNT = 3 #Trim all words that appears less than 3 times in the whole corpus

class Vocab:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token:"EOS"}
        self.word2count = {}
        self.num_words = 3 #Count for the special tokens

    def addSentence(self, sentence):
        """
        Add a sentence to the vocabulary
        """
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        """
        Add a word to the vocabulary
        """
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.word2count[word] = 1
            self.num_words += 1

    def trim(self, min_count):
        """
        Remove word from the vocabulary if they are below a threshold min_count
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if int(v) >= min_count:
                keep_words.append(k)
        
        print(f"Wiht a minimum count threshold of {min_count}, we kept {len(keep_words)} words out of {len(self.word2index)} total words, this is {(len(keep_words)*100)/len(self.word2index)}% of the vocabulary")

        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token:"EOS"}
        self.word2count = {}
        self.num_words = 3 #Count for the special tokens

        for word in keep_words:
            self.addWord(word)


def unicodeToAscii(s):
    """
    Convert unicode string to ASCII
    """
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalizeString(s):
    """
    Convert a string to all lowercase, trim, keep only letters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile, corpus_name):
    """
    Read the processed cornell data and returns the vocabulary object
    returns:
    vocab [object]
    """
    lines = open(datafile, encoding="utf-8").read().strip().split("\n")
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]
    vocab = Vocab(corpus_name)

    return vocab, pairs

def filterPair(p):
    """
    filter a single pair with the maximum sentence length allowed
    returns:
    [bool]
    """
    return len(p[0].split(" ")) < MAX_SENTENCE_LENGTH and len(p[1].split(" ")) < MAX_SENTENCE_LENGTH


def filterPairs(pairs):
    """
    filters all conversations pairs
    returns:
    [list]
    """
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus_name, datafile):
    """
    Process the data to get a vocabulary as well as the filtered conversation pairs
    returns:
    vocab [object]
    pairs [list]
    """

    logger.info("Creating the vocabulary...")
    vocab, pairs = readVocs(datafile, corpus_name)

    logger.info(f"Vocabulary created. There are {len(pairs)} pairs in total. Filtering the pairs now...")
    filtered_pairs = filterPairs(pairs)

    logger.info(f"There are {len(pairs)} after filtering. Counting the words now...")
    for pair in pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])
    
    logger.info(f"There are {vocab.num_words} word in the vocabulary")

    return vocab, filtered_pairs

### We choose here to filter out all pairs with trimmed words. We can also add an unknown token and keep all the pairs
def trimRareWords(vocab, pairs):
    vocab.trim(MIN_COUNT)
    keep_pairs = []

    for pair in pairs:
        convLine1, keep1 = pair[0], True
        convLine2, keep2 = pair[1], True

        for word in convLine1.split(" "):
            if word not in vocab.word2index:
                keep1 = False
                break

        if keep1:
            for word in convLine2.split(" "):
                if word not in vocab.word2index:
                    keep2 = False
                    break
        
        if keep1 and keep2:
            keep_pairs.append(pair)
    
    logger.info(f"From a total of {len(pairs)} sentence pairs, we trimmed {len(pairs)-len(keep_pairs)} of them, so {((len(pairs)-len(keep_pairs))*100)/len(pairs)}% of the original amount")
    return keep_pairs

if __name__ == "__main__":
    corpus_name = "cornell"
    datafile = "data/processed/cornell_movie/processed_movie_lines.txt"
    save_file = "data/processed/cornell_movie/filtered_conversation_pairs.txt"
    vocab, pairs = loadPrepareData(corpus_name, datafile)
    filtered_pairs = trimRareWords(vocab, pairs)


    delimiter = "\t"
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    with open(save_file, "w", encoding="utf-8") as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator="\n")
        for pair in filtered_pairs:
            writer.writerow(pair)
