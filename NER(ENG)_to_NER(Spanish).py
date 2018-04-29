import cnn_rnn
import lasagne
import numpy as np
import argparse
import sample

parser = argparse.ArgumentParser()
parser.add_argument('--labeling_rates', nargs='+', type=float)

args = parser.parse_args()

TASKS = args.tasks
LABELING_RATES = args.labeling_rates

print('LABELING_RATES', LABELING_RATES)

MIN_PERIODS = [4, 100] 
MAX_ITER = 1000000 
EXITS = [False, False]
USE_DEV = True

if __name__ == '__main__':
    char_set = set()
    for it in [1,2]:
        if it==1:
            task = 'ner_span'
        else:
            task = 'ner'
            
        t = __import__(task)
        data_list = [t.TRAIN_DATA, t.DEV_DATA, t.TEST_DATA]
        char_index, _ = t.create_char_index(data_list)
        for k, v in char_index.iteritems():
            char_set.add(k)
    char_index, char_cnt = {}, 0
    for char in char_set:
        char_index[char] = char_cnt
        char_cnt += 1

    models, eval_funcs = [], []
    for i in [0,1]:
        if i==0:
            task = 'ner_span'
        else:
            task = 'ner'
            
        t = __import__(task)
        word_index, word_cnt = t.create_word_index([t.TRAIN_DATA, t.DEV_DATA, t.TEST_DATA])
        wx, y, m = t.read_data(t.TRAIN_DATA, word_index)
        if USE_DEV and task == 'ner':
            dev_wx, dev_y, dev_m = t.read_data(t.TEST_DATA, word_index)
            wx, y, m = np.vstack((wx, dev_wx)), np.vstack((y, dev_y)), np.vstack((m, dev_m))
        twx, ty, tm = t.read_data(t.DEV_DATA, word_index)
        x, cm = t.read_char_data(t.TRAIN_DATA, char_index)
        if USE_DEV and task == 'ner':
            dev_x, dev_cm = t.read_char_data(t.TEST_DATA, char_index)
            x, cm = np.vstack((x, dev_x)), np.vstack((cm, dev_cm))
        tx, tcm = t.read_char_data(t.DEV_DATA, char_index)
        if task == 'ner':
            list_prefix = t.read_list()
            gaze = t.read_list_data(t.TRAIN_DATA, list_prefix)
            tgaze = t.read_list_data(t.DEV_DATA, list_prefix)
            if USE_DEV:
                dev_gaze = t.read_list_data(t.TEST_DATA, list_prefix)
                gaze = np.vstack((gaze, dev_gaze))
        else:
            gaze, tgaze = None, None
        
        model = cnn_rnn.cnn_rnn(char_cnt, len(t.LABEL_INDEX), word_cnt)
        model.min_epoch = MIN_PERIODS[i]

        if task == 'ner_span':
            model.w_embedding_size = 64
        else:
            model.w_embedding_size = 50
        
        model.joint = True

        if LABELING_RATES[i] < 1.0:
            ind = sample.create_sample_index(LABELING_RATES[i], x.shape[0])
            x, y, m, wx, cm, gaze = sample.sample_arrays((x, y, m, wx, cm, gaze), ind)
        model.add_data(x, y, m, wx, cm, gaze, tx, ty, tm, twx, tcm, tgaze)
        model.build()
        if task == 'ner_span':
            words, embeddings = t.read_word2embedding(t.PKL_FILE)
            model.set_embedding_pkl(words, embeddings, word_index, lower=False)
        else:
            word2embedding = t.read_word2embedding()
            model.set_embedding(word2embedding, word_index)
        model.step_train_init()
        models.append(model)
        eval_funcs.append(t.evaluate)

    prev_params = None
    max_f1s = [0.0, 0.0, 0.0]
    print "\t".join(['task', 'epoch', 'iter', 'max_f1', 'f1', 'prec', 'recall'])
    iter = 0
    while True:
        for i in range(len(models)):
            model = models[i]
            if prev_params is not None and iter < MAX_ITER:
                lasagne.layers.set_all_param_values(model.char_layer, prev_params)
                if cnn_rnn.REDUCE:
                    model.trans.set_value(prev_trans)
            if iter >= MAX_ITER and EXITS[i]:
                py = None
            else:
                py = model.step_train()
            if py is not None:
                iter += 1
                acc, f1, prec, recall = eval_funcs[i](py, model.ty, model.tm, full = True)
                # if f1 > max_f1s[i] or iter == 9 or iter == 10:
                #     model.store_params(0, '{}.{}.{}.params'.format(TASKS[i], iter, model.iter))
                max_f1s[i] = max(max_f1s[i], f1)
                print TASKS[i], model.epoch, model.iter, max_f1s[i], f1, prec, recall
            if iter < MAX_ITER:
                prev_params = lasagne.layers.get_all_param_values(model.char_layer)
                if cnn_rnn.REDUCE:
                    prev_trans = model.trans.get_value()
