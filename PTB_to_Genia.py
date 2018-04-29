import cnn_rnn
import lasagne
import sample
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--labeling_rates', nargs='+', type=float)

args = parser.parse_args()

LABELING_RATES = args.labeling_rates
VERY_TOP_JOINT = True

print('LABELING_RATES', LABELING_RATES)
print('VERY_TOP_JOINT', VERY_TOP_JOINT)

MIN_PERIODS = [4, 100]
EXITS = [False, False]
MAX_ITER = 1000000
USE_DEV = True

if __name__ == '__main__':
    char_set, word_set = set(), set()
    for it in [1,2]:
        if it==1:
            task = 'genia'
        else:
            task = 'ptb'
    
        t = __import__(task)
        data_list = [t.TRAIN_DATA, t.DEV_DATA]       
        data_list.append(t.TEST_DATA)
        
        char_index, _ = t.create_char_index(data_list)
        for k, v in char_index.iteritems():
            char_set.add(k)
        word_index, _ = t.create_word_index(data_list)
        for k, v in word_index.iteritems():
            word_set.add(k)
    char_index, char_cnt = {}, 0
    for char in char_set:
        char_index[char] = char_cnt
        char_cnt += 1
    word_index, word_cnt = {}, 0
    for word in word_set:
        word_index[word] = word_cnt
        word_cnt += 1
    
    models, eval_funcs = [], []
    for i in [0,1]:
        if i==0:
            task = 'genia'
        else:
            task = 'ptb'
            
        t = __import__(task)
        wx, y, m = t.read_data(t.TRAIN_DATA, word_index)
        
        dev_wx, dev_y, dev_m = t.read_data(t.TEST_DATA, word_index)
        wx, y, m = np.vstack((wx, dev_wx)), np.vstack((y, dev_y)), np.vstack((m, dev_m))
        
        twx, ty, tm = t.read_data(t.DEV_DATA, word_index)
        x, cm = t.read_char_data(t.TRAIN_DATA, char_index)
        dev_x, dev_cm = t.read_char_data(t.TEST_DATA, char_index)
        x, cm = np.vstack((x, dev_x)), np.vstack((cm, dev_cm))
        
        tx, tcm = t.read_char_data(t.DEV_DATA, char_index)
              
        gaze, tgaze = None, None
        
        model = cnn_rnn.cnn_rnn(char_cnt, len(t.LABEL_INDEX), word_cnt)
        model.min_epoch = MIN_PERIODS[i]
        
        model.very_top_joint = True
   
        if LABELING_RATES[i] < 1.0:
            ind = sample.create_sample_index(LABELING_RATES[i], x.shape[0])
            x, y, m, wx, cm, gaze = sample.sample_arrays((x, y, m, wx, cm, gaze), ind)
        
        model.add_data(x, y, m, wx, cm, gaze, tx, ty, tm, twx, tcm, tgaze)
        model.build()
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
            if iter >= MAX_ITER and EXITS[i]:
                py = None
            else:
                py = model.step_train()
            if py is not None:
                iter += 1
                acc, f1, prec, recall = eval_funcs[i](py, model.ty, model.tm, full = True)
                max_f1s[i] = max(max_f1s[i], f1)
                print TASKS[i], model.epoch, model.iter, max_f1s[i], f1, prec, recall
            if iter < MAX_ITER:
                prev_params = lasagne.layers.get_all_param_values(model.char_layer)
