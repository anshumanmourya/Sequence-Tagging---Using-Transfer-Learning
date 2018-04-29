
import numpy as np
from collections import defaultdict as dd
from scipy import sparse as sp
import cnn_rnn
import sample

LABEL_INDEX = ['PRP$', 'VBG', 'VBD', '``', 'VBN', 'POS', "''", 'VBP', 'WDT', 'JJ',\
 'WP', 'VBZ', 'DT', '#', 'RP', '$', 'NN', 'FW', ',', '.', 'TO', 'PRP', 'RB', '-LRB-',\
  ':', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'LS', 'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN', 'WP$',\
   'MD', 'NNPS', '-RRB-', 'JJS', 'JJR', 'SYM', 'UH']

MAX_LEN = 251
MAX_CHAR_LEN = 17 # 17

DIR = 'pos_tree/'
TRAIN_DATA = DIR + 'train.txt'
DEV_DATA = DIR + 'test.txt'
TEST_DATA = DIR + 'dev.txt'

HASH_FILE = 'words.lst'
EMB_FILE = 'embeddings.txt'

LIST_FILE = 'eng.list'

RARE_WORD = False
RARE_CHAR = False

USE_DEV = True # False
LABELING_RATE = 1.0 # 1.0

def read_list():
    prefix = dd(list)
    for line in open(LIST_FILE):
        inputs = line.strip().split()
        cat = inputs[0]
        prefix[inputs[1]].append((" ".join(inputs[1:]), cat))
    for k in prefix.keys():
        prefix[k].sort(key = lambda x: len(x[0]), reverse = True)
    return prefix

def label_decode(label):
    if label == 'O':
        return 'O', 'O'
    return tuple(label.split('-'))

def process_labels(y, m):
    def old_match(y_prev, y_next):
        l_prev, l_next = LABEL_INDEX[y_prev], LABEL_INDEX[y_next]
        c1_prev, c2_prev = label_decode(l_prev)
        c1_next, c2_next = label_decode(l_next)
        if c2_prev != c2_next: return False
        if c1_next == 'B': return False
        return True

    ret = np.zeros(y.shape, dtype = y.dtype)
    for i in range(y.shape[0]):
        j = 0
        while j < y.shape[1]:
            if m[i, j] == 0: break
            if y[i, j] == 0:
                j += 1
                continue
            k = j + 1
            while k < y.shape[1] and old_match(y[i, j], y[i, k]):
                k += 1
            _, c2 = label_decode(LABEL_INDEX[y[i, j]])
            if k - j == 1:
                ret[i, j] = LABEL_INDEX.index('S-{}'.format(c2))
            else:
                ret[i, j] = LABEL_INDEX.index('B-{}'.format(c2))
                ret[i, k - 1] = LABEL_INDEX.index('E-{}'.format(c2))
                for p in range(j + 1, k - 1):
                    ret[i, p] = LABEL_INDEX.index('I-{}'.format(c2))
            j = k
    return ret

def process(word):
    word = word.lower()
    word = "".join(c if not c.isdigit() else '0' for c in word)
    return word

def cnt_line(filename):
    line_cnt = 0
    cur_flag = False
    for line in open(filename):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if cur_flag:
                line_cnt += 1
            cur_flag = False
            continue
        cur_flag = True
    if cur_flag: line_cnt += 1
    return line_cnt

def create_word_index(filenames):
    word_index, word_cnt = {}, 1

    if RARE_WORD:
        word_cnt += 1
        word_stats = dd(int)
        for filename in filenames:
            for line in open(filename):
                inputs = line.strip().split()
                if len(inputs) < 2: continue
                word = inputs[0]
                word = process(word)
                word_stats[word] += 1
        single_words = []
        for word, cnt in word_stats.iteritems():
            if cnt == 1:
                single_words.append(word)
        single_words = set(single_words)

    for sign, filename in enumerate(filenames):
        for line in open(filename):
            inputs = line.strip().split()
            if len(inputs) < 2: continue
            word = inputs[0]
            word = process(word)
            if RARE_WORD and word in single_words:
                word_index[word] = 1
                continue
            if word in word_index: continue
            word_index[word] = word_cnt
            word_cnt += 1
    return word_index, word_cnt

def create_char_index(filenames):
    char_index, char_cnt = {}, 3

    if RARE_CHAR:
        char_cnt += 1
        char_stats = dd(int)
        for filename in filenames:
            for line in open(filename):
                inputs = line.strip().split()
                if len(inputs) < 2: continue
                for c in inputs[0]:
                    char_stats[c] += 1
        rare_chars = []
        for char, cnt in char_stats.iteritems():
            if cnt < 100:
                rare_chars.append(char)
        rare_chars = set(rare_chars)

    for filename in filenames:
        for line in open(filename):
            inputs = line.strip().split()
            if len(inputs) < 2: continue
            for c in inputs[0]:
                if RARE_CHAR and c in rare_chars:
                    char_index[c] = 3
                    continue
                if c not in char_index:
                    char_index[c] = char_cnt
                    char_cnt += 1
    return char_index, char_cnt

def read_data(filename, word_index):
    line_cnt = cnt_line(filename)
    x, y = np.zeros((line_cnt, MAX_LEN), dtype = np.int32), np.zeros((line_cnt, MAX_LEN), dtype = np.int32)
    mask = np.zeros((line_cnt, MAX_LEN), dtype = np.float32)
    i, j = 0, 0
    for line in open(filename):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if j > 0:
                i, j = i + 1, 0
            continue
        word, label = inputs[0], inputs[-1]
        word = process(word)
        word_ind, label_ind = word_index[word], LABEL_INDEX.index(label)
        x[i, j] = word_ind
        y[i, j] = label_ind
        mask[i, j] = 1.0
        j += 1
    # y = process_labels(y, mask)
    return x, y, mask

def read_pos_data(filename, pos_index, which):
    line_cnt = cnt_line(filename)
    x = np.zeros((line_cnt, MAX_LEN), dtype = np.int32)
    i, j = 0, 0
    for line in open(filename):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if j > 0:
                i, j = i + 1, 0
            continue
        pos = inputs[which]
        x[i, j] = pos_index[pos]
        j += 1
    return x

def write_to_file(output_file, input_file, py):
    i, j = 0, 0
    fout = open(output_file, 'w')
    for line in open(input_file):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if j > 0:
                i, j = i + 1, 0
                fout.write("\n")
            continue
        fout.write(line.strip() + " " + LABEL_INDEX[py[i, j]] + "\n")
        j += 1
    fout.close()

def read_char_data(filename, char_index):
    line_cnt = cnt_line(filename)
    x = np.zeros((line_cnt, MAX_LEN, MAX_CHAR_LEN), dtype = np.int32)
    mask = np.zeros((line_cnt, MAX_LEN, MAX_CHAR_LEN), dtype = np.float32)
    i, j = 0, 0
    for line in open(filename):
        inputs = line.strip().split()
        if len(inputs) < 2:
            if j > 0:
                i, j = i + 1, 0
            continue
        word, label = inputs[0], inputs[-1]
        for k, c in enumerate(word):
            if k + 1 >= MAX_CHAR_LEN: break
            x[i, j, k + 1] = char_index[c]
            mask[i, j, k + 1] = 1.0
        x[i, j, 0] = 1
        mask[i, j, 0] = 1.0
        if len(word) + 1 < MAX_CHAR_LEN:
            x[i, j, len(word) + 1] = 2
            mask[i, j, len(word) + 1] = 1.0
        j += 1
    return x, mask

def evaluate(py, y_, m_, full = False, ind2word = None, x = None):
    if len(py.shape) > 1:
        py = np.argmax(py, axis = 1)
    y, m = y_.flatten(), m_.flatten()
    acc = 1.0 * (np.array(y == py, dtype = np.int32) * m).sum() / m.sum()

    if ind2word is not None:
        fout = open('error.txt', 'w')
        temp_ind = (x * m_).flatten()[y != py]
        temp_y = y[y != py]
        temp_py = py[y != py]
        for i in range(temp_ind.shape[0]):
            ind = temp_ind[i]
            if ind not in ind2word: continue
            fout.write("{} {} {}\n".format(ind2word[ind], LABEL_INDEX[temp_y[i]], LABEL_INDEX[temp_py[i]]))
        fout.close()

    return acc, acc, acc, acc

def read_word2embedding():
    words = []
    for line in open(HASH_FILE):
        words.append(line.strip())
    word2embedding = {}
    for i, line in enumerate(open(EMB_FILE)):
        if words[i] in word2embedding: continue
        inputs = line.strip().split()
        word2embedding[words[i]] = np.array([float(e) for e in inputs], dtype = np.float32)
    return word2embedding


