import numpy as np
import re
import sys
import random
import json
import collections
import logging
import six
#import parameters as params
import ocnli.mnli_code.util.parameters as params
import pickle
import tensorflow as tf

FIXED_PARAMETERS = params.load_parameters()

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

logging.basicConfig(level=logging.INFO)

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()

def tokenize_zh(string):
    string = re.sub(r'\(|\)', '', string)
    return [c.strip() for c in string.strip()]
    #return string.split()

def load_nli_data(path, snli=False,partial_input=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    logging.info('Loading data: %s, snli=%s, partial_input=%s' % (path,str(snli),str(partial_input)))
    
    data = []
    total = 0
    collected = 0
    #lines = tf.gfile.Open(path, "r")
    with tf.gfile.Open(path, "r") as f:
        for k,line in enumerate(f):
            loaded_example = json.loads(line)
            total += 1
            if partial_input:
                loaded_example['sentence1'] = "####"
            
            if "gold_label" not in loaded_example:
                loaded_example["gold_label"] = loaded_example["label"]
            if "sentence1_binary_parse" not in loaded_example:
                sentence1 = tokenize_zh(convert_to_unicode(loaded_example['sentence1']))
                sentence2 = tokenize_zh(convert_to_unicode(loaded_example['sentence2']))
                loaded_example["sentence1_binary_parse"] = ' '.join(sentence1)
                loaded_example["sentence2_binary_parse"] = ' '.join(sentence2)
                loaded_example['sentence1'] = ' '.join(sentence1)
                loaded_example['sentence2'] = ' '.join(sentence2)

            if loaded_example["gold_label"] not in LABEL_MAP: continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"

            ## log the first example jsut to sanity check
            if k == 0: logging.info(loaded_example)
            collected += 1
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)

    logging.info('Loaded %d / %d examples' % (collected,total))
    return data

def load_nli_data_genre(path, genre, snli=True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    #with open(path) as f:
    with tf.gfile.Open(path, "r") as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data



def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence1_binary_parse']))
            word_counter.update(tokenize(example['sentence2_binary_parse']))

    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS["seq_length"]), dtype=np.int32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS["seq_length"] - len(token_sequence)

                for i in range(FIXED_PARAMETERS["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                    else:
                        if token_sequence[i] in word_indices:
                            index = word_indices[token_sequence[i]]
                        else:
                            index = word_indices[UNKNOWN]
                    example[sentence + '_index_sequence'][i] = index


def loadEmbedding_zeros(path, word_indices):
    """
    Load GloVe embeddings. Initializng OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), FIXED_PARAMETERS["word_embedding_dim"]), dtype='float32')
    
    #with open(path, 'r') as f:
    with tf.gfile.Open(path, "r") as f: 
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def loadEmbedding_rand(path, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = FIXED_PARAMETERS["word_embedding_dim"]
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1,m), dtype="float32")
    total = 0
    total_words = set()

    #with open(path, 'r') as f:
    with tf.gfile.Open(path, "r") as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS["embeddings_to_load"] != None:
                if i >= FIXED_PARAMETERS["embeddings_to_load"]:
                    break
            s = line.split()
            word = convert_to_unicode(s[0])
            if word in word_indices:
                total_words.add(word)
                try: 
                    emb[word_indices[word], :] = np.asarray(s[1:])
                    total += 1
                except:
                    logging.warning("problematic vector for '%s'" % word)
    ##
    logging.info('Loaded vectors for %d words/tokens/characters (%d/%d)' % (total,len(total_words),len(word_indices)))
    return emb
