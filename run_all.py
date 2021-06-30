# author: Arven Sheinin
# adjusted from: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
from collections import defaultdict
import emoji
import random
import pickle
from bi_lstm_crf import *
import pandas as pd
import torch
# import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import emoji
import random
import time
# from math import floor, ceil
from datetime import timedelta
import pickle


# bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

torch.manual_seed(123)
random.seed(123)

# Helper functions 

def prep_data(PATH, labeled=True):
    data = []
    if labeled:
        with open(PATH) as train_raw:
            tup = ([], [])
            for iline, line in enumerate(train_raw.readlines()):
                split_line = line.split("\t")
                if len(split_line) == 2:
                    word, tag = split_line[0], split_line[1][0]
                    if word != emoji.demojize(word):
                        word = emoji.demojize(word).replace(":", "").split("_")[-1]
                    if word.startswith("*"): word = word[1:]
                    if word.endswith("*"): word = word[:-1]
                    if word.startswith("http://"): word = "+HYPERLINK+" 
                    # if word.startswith("#"): word = "*HASHTAG*"
                    if word.startswith("#"): word = word[1:]
                    if word.startswith("@"): word = "+MENTION+"
                    # if emoji.demojize(word)!=word: word = "*EMOJI*"
                    tup[0].append(word)
                    tup[1].append(tag.rstrip())
                else:
                    data.append(tup)
                    tup = ([], [])
            train_raw.close()
    else:
        with open(PATH) as test_raw:
            sentence = []
            for iline, line in enumerate(test_raw.readlines()):
                if line.rstrip():
                    word = line.split()[0].rstrip()
                    if word != emoji.demojize(word):
                        word = emoji.demojize(word).replace(":", "").split("_")[-1]
                    if word.startswith("*"): word = word[1:]
                    if word.endswith("*"): word = word[:-1]
                    word = emoji.demojize(word)
                    if word.startswith("http://"): word = "+HYPERLINK+" 
                    # if word.startswith("#"): word = "*HASHTAG*"
                    if word.startswith("#"): word = word[1:]
                    if word.startswith("@"): word = "+MENTION+"
                    # if emoji.demojize(word)!=word: word = "*EMOJI*"
                    sentence.append(word)
                else:
                    data.append(sentence)
                    sentence = []    
    return data


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    # idxs = [to_ix[w] for w in seq]
    idx = []
    for w in seq:
        try:
            idx.append(to_ix[w])
        except:
            try:
                idx.append(to_ix[handle_rare(w)])
            except:
                idx.append(0)
    return torch.tensor(idx, dtype=torch.long)


def handle_rare(s):
    if s == '-DOCSTART-':
        return s
    elif s.isupper():
        if '-' in s:
            return '+-Upper+'
        return '+Upper+'
    # if all uppercase
    elif s.isupper():
        if '-' in s:
            return '+-UPPER+'
        return '+UPPER+'
    #if hyphen in the string
    elif '-' in s:
        return '+-+'
    elif s.lower().endswith("ing"):
        return "+ING+"
    elif s.lower().endswith("ed"):
        return "+ED+"
    elif emoji.demojize(s)!=s:
        return s.replace(":", "").split("_")[-1]
    elif check_CD(s):
        return "+NUM+"
    return "+UNK+"


def check_CD(s):
    """
    Checks if the input string is a CD (cardinal number).
    """
    numbers = ['one','two','three','four','five','six','seven','eight','nine','zero',"eleven", "twelve", "thirteen","fourteen", "fifteen", "sixteen", "seventeen", "eighteen","nineteen"]
    tens = ["ten", "twenty", "thirty", "forty","fifty", "sixty", "seventy", "eighty", "ninety"]
    thousand = ["thousand", "million", "billion"]

    if s.lower() in numbers or s.lower() in tens or s.lower() in thousand:
        return True
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def create_unks(data, size=0.1):
    output = []# data.copy()
    for ipair, seq in enumerate(data):
        if random.random() < size:
            out = ([] ,[])
            words = seq[0]
            tags = seq[1]
            one_unk = False
            for i, pair in enumerate(zip(words, tags)):
                word = pair[0]
                tag = pair[1]
                if tag == "O" and random.random() < .1 and not one_unk:
                    word = "*UNK*"
                    one_unk = True
                out[0].append(word)
                out[1].append(tag)
            output.append(out)
    return output
                    
# Compute log sum exp in a numerically stable way for the forward algorithm

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, bi = True):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=bi)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq





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


with open("./my_data/training_data.pickle", "wb") as f:
    pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

with open("./my_data/dev_data.pickle", "wb") as f:
    pickle.dump(dev_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


vocab = defaultdict(int)
for pair in training_data:
    for word, tag in zip(pair[0], pair[1]):
        vocab[word.lower()] += 1
rare = 0 # count rare words

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


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 50 #int(sys.argv[1])
HIDDEN_DIM = 128 #int(sys.argv[2])
EPOCHS = 50 #int(sys.argv[3])
train_path = "./data/train/train.txt" #sys.argv[4]
dev_path = "./data/dev/dev.txt" #sys.argv[5]
bi = True


with open('./my_data/word_to_ix.pickle', 'rb') as handle:
    word_to_ix = pickle.load(handle)

with open('./my_data/tag_to_ix.pickle', 'rb') as handle:
    tag_to_ix = pickle.load(handle)

# with open(train_path, 'rb') as handle:
#     training_data = pickle.load(handle)

# with open(dev_path, 'rb') as handle:
#     dev_data = pickle.load(handle)

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


model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, bi=bi)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

lmbda = lambda epoch: 0.95
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
losses = []
val_losses = []
accuracies = []
val_accuracies = []
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(EPOCHS+1):  # again, normally you would NOT do 300 epochs, it is toy data
    model.train()
    start = time.time()
    total_loss = 0
    trues = 0
    half_train = int(len(training_data)/2)
    for i, (sentence, tags) in enumerate(training_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)
        total_loss += loss.item()
        avg_loss = total_loss/(i+1)
        
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

        predicted = model.forward(sentence_in)[1]
        if predicted == [tag_to_ix[t] for t in tags]: 
            trues += 1
        accuracy = trues/(i+1)
        # if i%100==0:
        #     print(f"Epoch {epoch+1}/{EPOCHS}; sentence {i+1}: loss = {avg_loss:.4f}; accuracy = {accuracy:.4f}")
    scheduler.step()
    # validation loop
    model.eval()
    val_trues = 0
    val_total_loss = 0
    half_dev = int(len(dev_data)/2)
    for i, (sentence, tags) in enumerate(dev_data):
        sentence_in = prepare_sequence(sentence, word_to_ix)
        indexed_tags = [tag_to_ix[t] for t in tags]
        targets = torch.tensor(indexed_tags, dtype=torch.long)
        
        val_loss = model.neg_log_likelihood(sentence_in, targets)
        val_total_loss += val_loss.item()
        val_avg_loss = val_total_loss/(i+1)
        predicted = model.forward(sentence_in)[1]
        if predicted == [tag_to_ix[t] for t in tags]: 
            val_trues += 1
        val_accuracy = val_trues/(i+1)
    
    val_losses.append(val_avg_loss)
    losses.append(avg_loss)
    val_accuracies.append(val_accuracy)
    accuracies.append(accuracy)
    print(f"epoch {epoch}/{EPOCHS}: training loss = {avg_loss:.4f};\ttraining accuracy = {accuracy:.4f};\t\
        validation loss = {val_avg_loss:.4f};\tvalidation accuracy = {val_accuracy:.4f}\t\
        time = {str(timedelta(seconds=time.time() - start))}")
    data_type = "" #"unk" if "unk" in sys.argv[5] else ""
    torch.save(model.state_dict(), f"./my_models/model_{EMBEDDING_DIM}_{HIDDEN_DIM}_{THRESHOLD}.pth")
# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!

loss_mat = pd.DataFrame.from_dict({
    "train loss": losses,
    "validation loss": val_losses
})



data_type = ""#"unk" if "unk" in sys.argv[5] else ""
pd.DataFrame.to_csv(loss_mat, f"./my_results/losses{EMBEDDING_DIM}_{HIDDEN_DIM}_{THRESHOLD}.csv")
torch.save(model.state_dict(), f"./my_models/model_{EMBEDDING_DIM}_{HIDDEN_DIM}_{THRESHOLD}.pth")