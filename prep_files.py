from collections import defaultdict
import emoji
import random
import pickle
from bi_lstm_crf import *


START_TAG = "<START>"
STOP_TAG = "<STOP>"
# EMBEDDING_DIM = 5
# HIDDEN_DIM = 4
THRESHOLD = 3

TRAIN_PATH = "./data/train/train.txt"
DEV_PATH = "./data/dev/dev.txt"

training_data = prep_data(TRAIN_PATH)
print(len(training_data))
additional_data = prep_data("./data/ext/mic-cis.txt")
more_data = prep_data("data/ext/gmb.txt")
training_data.extend(additional_data)
training_data.extend(more_data)
del additional_data
del more_data
print(len(training_data))
dev_data = prep_data(DEV_PATH)

print(f"data + dev length = {len(training_data) + len(dev_data)}")

# random.shuffle(data)
# training_data = data[:training_length]
# dev_data = data[training_length:]

with open("./my_data/training_data.pickle", "wb") as f:
    pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

with open("./my_data/dev_data.pickle", "wb") as f:
    pickle.dump(dev_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

# train_unks = create_unks(training_data, .1)
# dev_unks = create_unks(dev_data, .1)
# training_data.extend(train_unks)
# dev_data.extend(dev_unks)
# del train_unks
# del dev_unks

# with open("./my_data/training_data_unk.pickle", "wb") as f:
#     pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
#     f.close()

# with open("./my_data/dev_data_unk.pickle", "wb") as f:
#     pickle.dump(dev_data, f, protocol=pickle.HIGHEST_PROTOCOL)
#     f.close()

vocab = defaultdict(int)
for pair in training_data:
    for word, tag in zip(pair[0], pair[1]):
        vocab[word.lower()] += 1
rare = 0 # count rare words

# word_to_ix = {"*UNK*": 0}
# for sentence, tags in training_data:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)

word_to_ix = {"+UNK+": 0}
tag_to_ix = {"O": 0, "B": 1, "I": 2}
# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for iwrod, word in enumerate(sent):
        if vocab[word.lower()] < THRESHOLD: word = handle_rare(word)
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

with open("./my_data/word_to_ix.pickle", "wb") as f:
    pickle.dump(word_to_ix, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

with open("./my_data/tag_to_ix.pickle", "wb") as f:
    pickle.dump(tag_to_ix, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()