
import lasagne
import numpy as np
import theano.tensor as T
import theano
import cPickle
# from data import *
import sys
from theano import sparse

CRF_INIT = True
COST = True
COST_CONST = 5.0

PERIOD = 100 # 100, 4000
MIN_PERIOD = 4000

NOISE = False
NOISE_RATE = 0.1

REDUCE = False

def print_arr(y):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            print y[i, j],
        print ' '

def theano_logsumexp(x, axis=None):
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

class CRFLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_classes, W_sim = lasagne.init.GlorotUniform(), W = lasagne.init.GlorotUniform(), \
        W_init = lasagne.init.GlorotUniform(), mask_input = None, label_input = None, **kwargs):

        super(CRFLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.num_classes = num_classes
        self.W_sim = self.add_param(W_sim, (num_classes, num_classes))
        self.W = self.add_param(W, (num_inputs, num_classes))
        self.mask_input = mask_input
        self.label_input = label_input
        if CRF_INIT:
            self.W_init = self.add_param(W_init, (1, num_classes))
        else:
            self.W_init = None

    def get_output_shape_for(self, input_shape):
        return (1, )

    def get_output_for(self, input, **kwargs):
        def norm_fn(f, mask, label, previous, W_sim):
            # f: inst * class, mask: inst, previous: inst * class, W_sim: class * class
            next = previous.dimshuffle(0, 1, 'x') + f.dimshuffle(0, 'x', 1) + W_sim.dimshuffle('x', 0, 1)
            if COST:
                next = next + COST_CONST * (1.0 - T.extra_ops.to_one_hot(label, self.num_classes).dimshuffle(0, 'x', 1))
            # next: inst * prev * cur
            next = theano_logsumexp(next, axis = 1)
            # next: inst * class
            mask = mask.dimshuffle(0, 'x')
            next = previous * (1.0 - mask) + next * mask
            return next

        f = T.dot(input, self.W)
        # f: inst * time * class

        initial = f[:, 0, :]
        if CRF_INIT:
            initial = initial + self.W_init[0].dimshuffle('x', 0)
        if COST:
            initial = initial + COST_CONST * (1.0 - T.extra_ops.to_one_hot(self.label_input[:, 0], self.num_classes))
        outputs, _ = theano.scan(fn = norm_fn, \
         sequences = [f.dimshuffle(1, 0, 2)[1: ], self.mask_input.dimshuffle(1, 0)[1: ], self.label_input.dimshuffle(1, 0)[1:]], \
         outputs_info = initial, non_sequences = [self.W_sim], strict = True)
        norm = T.sum(theano_logsumexp(outputs[-1], axis = 1))

        f_pot = (f.reshape((-1, f.shape[-1]))[T.arange(f.shape[0] * f.shape[1]), self.label_input.flatten()] * self.mask_input.flatten()).sum()
        if CRF_INIT:
            f_pot += self.W_init[0][self.label_input[:, 0]].sum()

        labels = self.label_input
        # labels: inst * time
        shift_labels = T.roll(labels, -1, axis = 1)
        mask = self.mask_input
        # mask : inst * time
        shift_mask = T.roll(mask, -1, axis = 1)

        g_pot = (self.W_sim[labels.flatten(), shift_labels.flatten()] * mask.flatten() * shift_mask.flatten()).sum()

        return - (f_pot + g_pot - norm) / f.shape[0]

class CRFDecodeLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_classes, W_sim = lasagne.init.GlorotUniform(), W = lasagne.init.GlorotUniform(), \
        W_init = lasagne.init.GlorotUniform(), mask_input = None, **kwargs):

        super(CRFDecodeLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.W_sim = self.add_param(W_sim, (num_classes, num_classes))
        self.W = self.add_param(W, (num_inputs, num_classes))
        self.mask_input = mask_input
        if CRF_INIT:
            self.W_init = self.add_param(W_init, (1, num_classes))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_output_for(self, input, **kwargs):
        def max_fn(f, mask, prev_score, prev_back, W_sim):
            next_score = prev_score.dimshuffle(0, 1, 'x') + f.dimshuffle(0, 'x', 1) + W_sim.dimshuffle('x', 0, 1)
            next_back = T.argmax(next_score, axis = 1)
            next_score = T.max(next_score, axis = 1)
            mask = mask.dimshuffle(0, 'x')
            next_score = next_score * mask + prev_score * (1.0 - mask)
            next_back = next_back * mask + prev_back * (1.0 - mask)
            next_back = T.cast(next_back, 'int32')
            return [next_score, next_back]

        def produce_fn(back, mask, prev_py):
            # back: inst * class, prev_py: inst, mask: inst
            next_py = back[T.arange(prev_py.shape[0]), prev_py]
            next_py = mask * next_py + (1.0 - mask) * prev_py
            next_py = T.cast(next_py, 'int32')
            return next_py

        f = T.dot(input, self.W)

        init_score, init_back = f[:, 0, :], T.zeros_like(f[:, 0, :], dtype = 'int32')
        if CRF_INIT:
            init_score = init_score + self.W_init[0].dimshuffle('x', 0)
        ([scores, backs], _) = theano.scan(fn = max_fn, \
            sequences = [f.dimshuffle(1, 0, 2)[1: ], self.mask_input.dimshuffle(1, 0)[1: ]], \
            outputs_info = [init_score, init_back], non_sequences = [self.W_sim], strict = True)

        init_py = T.argmax(scores[-1], axis = 1)
        init_py = T.cast(init_py, 'int32')
        # init_py: inst, backs: time * inst * class
        pys, _ = theano.scan(fn = produce_fn, \
            sequences = [backs, self.mask_input.dimshuffle(1, 0)[1:]], outputs_info = [init_py], go_backwards = True)
        # pys: (rev_time - 1) * inst
        pys = pys.dimshuffle(1, 0)[:, :: -1]
        # pys : inst * (time - 1)
        return T.concatenate([pys, init_py.dimshuffle(0, 'x')], axis = 1)

class ElemwiseMergeLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, merge_function, **kwargs):
        super(ElemwiseMergeLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output

class cnn_rnn:

    def __init__(self, char_cnt, label_cnt, word_cnt, ind2word = None, lemma_cnt = 0, pos_cnt = 0):
        lasagne.random.set_rng(np.random)
        np.random.seed(13) # 13

        self.char_cnt = char_cnt
        self.label_cnt = label_cnt
        self.embedding_size = 25 # 25, 50
        self.hidden_size = 300 # 300
        self.learning_rate = 1e-2 # 1e-2
        self.batch_size = 10
        self.f_sizes = [3, 4, 5]
        self.filter_num = 200
        self.char_f_sizes = [3, 4, 5]
        self.char_filter_num = 20
        self.word_cnt = word_cnt
        self.w_embedding_size = 50 # 50, 64, 300 # 100?
        self.epoch = 100
        self.char_hidden_size = 80 # 80, 100
        self.dropout_rate = 0.5
        self.gaze_embedding_size = 7
        self.ind2word = ind2word
        self.test_batch_size = 100

        self.use_crf = True
        self.dropout = False
        self.double_layer = True
        self.char_double_layer = True
        self.use_gaze = True # True

        self.char_cnn = False
        self.char_rnn = True # true
        self.word_cnn = False
        self.word_embedding = True # true

        self.use_lemma = False # true
        self.lemma_cnt = lemma_cnt
        self.lemma_size = 10

        self.use_pos = False # true
        self.pos_cnt = pos_cnt
        self.pos_size = 10

        self.reduce_size = 15

        self.tanh = False # False
        self.tanh_size = 150 # 150
        self.joint = False # False
        self.top_joint = False # False
        self.very_top_joint = False

        self.min_epoch = 2

    def add_data(self, x, y, m, wx, cm, gaze, tx, ty, tm, twx, tcm, tgaze, lemma = None, tlemma = None, pos = None, tpos = None):
        self.x, self.y, self.m, self.wx, self.tx, self.ty, self.tm, self.twx = x, y, m, wx, tx, ty, tm, twx
        self.cm, self.tcm = cm, tcm
        self.gaze, self.tgaze = gaze, tgaze
        if self.gaze is None:
            self.use_gaze = False
        self.lemma, self.tlemma = lemma, tlemma
        if self.lemma is None:
            self.use_lemma = False
        self.pos, self.tpos = pos, tpos
        if self.pos is None:
            self.use_pos = False

    def build(self):
        x_sym = T.itensor3('x')
        y_sym = T.imatrix('y')
        m_sym = T.matrix('mask')
        wx_sym = T.imatrix('wx')
        lemma_sym = T.imatrix('lemma')
        pos_sym = T.imatrix('pos')
        cm_sym = T.tensor3('cmask')
        gaze_sym = T.tensor3('gaze')

        l = lasagne.layers.InputLayer(shape = (None, self.x.shape[1], self.x.shape[2]), input_var = x_sym)
        l = lasagne.layers.EmbeddingLayer(l, self.char_cnt, self.embedding_size)
        l_char = lasagne.layers.ReshapeLayer(l, (-1, [2], [3]))
        l_c_mask = lasagne.layers.InputLayer(shape = (None, self.x.shape[1], self.x.shape[2]), input_var = cm_sym)
        l_c_mask = lasagne.layers.ReshapeLayer(l_c_mask, (-1, [2]))
        l_gru = lasagne.layers.GRULayer(l_char, self.char_hidden_size, mask_input = l_c_mask)
        l_gru_2 = lasagne.layers.GRULayer(l_char, self.char_hidden_size, mask_input = l_c_mask, backwards = True)
        if self.char_double_layer:
            l_char_rnn = lasagne.layers.ConcatLayer([l_gru, l_gru_2], axis = 2)
            l_gru = lasagne.layers.GRULayer(l_char_rnn, self.char_hidden_size, mask_input = l_c_mask)
            l_gru_2 = lasagne.layers.GRULayer(l_char_rnn, self.char_hidden_size, mask_input = l_c_mask, backwards = True)
        l_gru = lasagne.layers.ReshapeLayer(l_gru, (-1, self.x.shape[1], [1], [2]))
        l_gru = lasagne.layers.SliceLayer(l_gru, -1, axis = 2)
        l_gru_2 = lasagne.layers.ReshapeLayer(l_gru_2, (-1, self.x.shape[1], [1], [2]))
        l_gru_2 = lasagne.layers.SliceLayer(l_gru_2, 0, axis = 2)
        l = lasagne.layers.ReshapeLayer(l, (-1, [3], [2]))
        ls = []
        l_c_mask = lasagne.layers.DimshuffleLayer(l_c_mask, (0, 'x', 1))
        l_filter = ElemwiseMergeLayer([l, l_c_mask], T.mul)
        for f_size in self.char_f_sizes:
            ls.append(lasagne.layers.Conv1DLayer(l_filter, self.char_filter_num, f_size))
        for i in range(len(ls)):
            ls[i] = lasagne.layers.GlobalPoolLayer(ls[i], T.max)
        l_c = lasagne.layers.ConcatLayer(ls)
        l_c = lasagne.layers.ReshapeLayer(l_c, (-1, self.x.shape[1], [1]))
        l_word = lasagne.layers.InputLayer(shape = (None, self.x.shape[1]), input_var = wx_sym)
        l_word = lasagne.layers.EmbeddingLayer(l_word, self.word_cnt, self.w_embedding_size, W = lasagne.init.Normal(std = 1e-3))
        self.embedding = l_word.W
        layer_list = []
        if self.word_embedding:
            layer_list.append(l_word)
        if self.char_rnn:
            if self.joint:
                l_grus = lasagne.layers.ConcatLayer([l_gru, l_gru_2], axis = 2)
                self.char_layer = l_grus
                layer_list.append(l_grus)
                char_output = lasagne.layers.get_output(self.char_layer)
                self.char_fn = theano.function([x_sym, cm_sym], char_output, on_unused_input = 'ignore')
            else:
                layer_list += [l_gru, l_gru_2]

        if len(layer_list) > 1:
            l = lasagne.layers.ConcatLayer(layer_list, axis = 2)
        else:
            l = layer_list[0]
        if not self.joint:
            self.char_layer = l

        l_mask = lasagne.layers.InputLayer(shape = (None, self.x.shape[1]), input_var = m_sym)
        ls = []
        l_1 = lasagne.layers.GRULayer(l, self.hidden_size, mask_input = l_mask)
        l = lasagne.layers.GRULayer(l, self.hidden_size, mask_input = l_mask, backwards = True)
        if self.double_layer:
            l = lasagne.layers.ConcatLayer([l_1, l], axis = 2)
            l_1 = lasagne.layers.GRULayer(l, self.hidden_size, mask_input = l_mask)
            l = lasagne.layers.GRULayer(l, self.hidden_size, mask_input = l_mask, backwards = True)
        layer_list = [l_1, l]
        if self.use_gaze:
            l_gaze = lasagne.layers.InputLayer(shape = (None, self.x.shape[1], self.label_cnt), input_var = gaze_sym)
            layer_list.append(l_gaze)
        l = lasagne.layers.ConcatLayer(layer_list, axis = 2)

        if self.top_joint:
            self.char_layer = l

        if self.use_crf:
            l_train = CRFLayer(l, self.label_cnt, mask_input = m_sym, label_input = y_sym)
            l_test = CRFDecodeLayer(l, self.label_cnt, mask_input = m_sym, W = l_train.W, W_sim = l_train.W_sim,\
             W_init = l_train.W_init)
            self.l = l_train
            if self.very_top_joint:
                self.char_layer = self.l

            loss = lasagne.layers.get_output(l_train)
            params = lasagne.layers.get_all_params(self.l, trainable = True)
            updates = lasagne.updates.adagrad(loss, params, learning_rate = self.learning_rate)
            self.train_fn = theano.function([x_sym, y_sym, m_sym, wx_sym, cm_sym, gaze_sym, lemma_sym, pos_sym], loss, updates = updates, on_unused_input = 'ignore')

            py = lasagne.layers.get_output(l_test, deterministic = True)
            self.test_fn = theano.function([x_sym, m_sym, wx_sym, cm_sym, gaze_sym, lemma_sym, pos_sym], py, on_unused_input = 'ignore')

    def store_params(self, iter, filename = None):
        if filename is None:
            fout = open('{}.{}'.format(self.__class__.__name__, iter), 'w')
        else:
            fout = open(filename, 'w')
        params = lasagne.layers.get_all_param_values(self.l)
        cPickle.dump(params, fout, cPickle.HIGHEST_PROTOCOL)
        fout.close()

    def load_params(self, filename):
        fin = open(filename)
        params = cPickle.load(fin)
        lasagne.layers.set_all_param_values(self.l, params)
        fin.close()

    def predict(self, tx, tm, twx, tcm, tgaze, tlemma = None, tpos = None):
        i = 0
        pys = []
        while i < self.tx.shape[0]:
            # j = min(self.x.shape[0], i + self.test_batch_size)
            j = i + self.test_batch_size
            s_x, s_m, s_wx, s_cm = tx[i: j], tm[i: j], twx[i: j], tcm[i: j]
            s_gaze = tgaze[i: j] if self.use_gaze else None
            s_lemma = tlemma[i: j] if self.use_lemma else None
            s_pos = tpos[i: j] if self.use_pos else None
            pys.append(self.test_fn(s_x, s_m, s_wx, s_cm, s_gaze, s_lemma, s_pos))
            i = j
        py = np.vstack(tuple(pys))
        if self.use_crf:
            return py.flatten()
        else:
            return py.argmax(axis = 1)

    def set_embedding(self, word2embedding, word_index):
        t_embedding = self.embedding.get_value()
        emb_hit_cnt = 0
        for word, embedding in word2embedding.iteritems():
            if word not in word_index: continue
            emb_hit_cnt += 1
            ind = word_index[word]
            t_embedding[ind, : embedding.shape[0]] = embedding
        print 'emb_hit_cnt', emb_hit_cnt
        self.embedding.set_value(t_embedding)


    def set_embedding_pkl(self, words, embeddings, word_index, lower = True):
        t_embedding = self.embedding.get_value()
        emb_words = set()
        for i in range(len(words)):
            word = words[i].lower() if lower else words[i]
            if word not in word_index: continue
            emb_words.add(word)
            ind = word_index[word]
            embedding = embeddings[i]
            t_embedding[ind, : embedding.shape[0]] = embedding
        print 'emb_hit_cnt', len(emb_words)
        self.embedding.set_value(t_embedding)


    def step_train_init(self):
        self.epoch = 0
        self.ind = np.random.permutation(self.x.shape[0])
        self.i = 0
        self.iter = 0

    def step_train(self):
        i = self.i
        j = min(self.x.shape[0], i + self.batch_size)
        ind = self.ind
        s_x, s_y, s_m, s_wx = self.x[ind[i: j]], self.y[ind[i: j]], self.m[ind[i: j]], self.wx[ind[i: j]]
        s_cm = self.cm[ind[i: j]]
        s_gaze = self.gaze[ind[i: j]] if self.use_gaze else None
        s_lemma = self.lemma[ind[i: j]] if self.use_lemma else None
        s_pos = self.pos[ind[i: j]] if self.use_pos else None
        if self.x.shape[0] > 0:
            loss = self.train_fn(s_x, s_y, s_m, s_wx, s_cm, s_gaze, s_lemma, s_pos)
        self.iter += 1

        if j == self.x.shape[0]:
            self.i = 0
            self.epoch += 1
            self.iter = 0
            self.ind = np.random.permutation(self.x.shape[0])
        else:
            self.i = j

        period = MIN_PERIOD if self.epoch <= self.min_epoch else PERIOD
        ret_flag = self.iter * self.batch_size % period == 0

        if ret_flag:
            return self.predict(self.tx, self.tm, self.twx, self.tcm, self.tgaze, self.tlemma, self.tpos)
        else:
            return None

    def step_predict(self):
        return self.predict(self.tx, self.tm, self.twx, self.tcm, self.tgaze, self.tlemma, self.tpos)

