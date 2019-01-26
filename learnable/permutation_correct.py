# -*- coding: UTF-8 -*-
import os
import codecs
import json
import random
import csv
import numpy as np
import argparse
import cPickle as pickle 
import editdistance

err_type = [u'del', u'ins', u'rpl']

data_size = 30000
dev_p = 0.2

file_train = "autocorrection_train_fix.json"
file_test_in = "autocorrection_sample_testing.json"
file_test_gt = "autocorrection_sample_testing_gt.json"
file_vocab = "vocab.txt"

vocab = {}
settrie = set()
group2word = {}
word2group = {}
data = {}
test_data = {}
available_words = []

rare_data = {}
done_data = {}

def segmentation(s):
    global settrie
    res = []
    l = len(s)
    i = 0
    while i < l:
        seg = s[i]
        if seg not in settrie:
            res.append('@<other>')
        else:
            tryseg = seg
            j = i
            while j + 1 < l and tryseg + s[j+1] in settrie:
                tryseg += s[j+1]
                j += 1
            seg_len = len(tryseg)
            while seg_len > 0:
                if tryseg[:seg_len] in vocab:
                    seg = tryseg[:seg_len]
                    break
                seg_len -= 1
            i += seg_len - 1
            if seg_len == 0:
                print "Something wrong 1"
            else:
                res.append(seg)
        i += 1
    return res

def get_bracket_pair(segs, pos):
    st = 1
    i = pos + 1
    l = len(segs)
    while i < l:
        curr_tag = segs[i]
        if curr_tag == u'{':
            st += 1
        elif curr_tag == u'}':
            st -= 1
            if st == 0:
                break
        i += 1
    if st > 0:
        return -1
    else:
        return i

def check_bracket_pair(segs, pos):
    right_pos = get_bracket_pair(segs, pos)
    r = syncheck(segs[pos+1:right_pos])
    if r == False:
        return -1
    return right_pos

def syncheck(segs):
    l = len(segs)
    i = 0
    while i < l:
        tag = segs[i]
        if tag == u'{':
            r = check_bracket_pair(segs, i)
            if r == -1:
                return False
            i = r
        elif tag == u'}':
            return False
        elif vocab[tag] == 0:
            pass
        elif vocab[tag] == 1:
            if i < l - 1 and segs[i+1] == u'{':
                r = check_bracket_pair(segs, i+1)
                if r == -1:
                    return False
                i = r
            else:
                return False
        elif vocab[tag] == 2:
            if i < l - 1 and segs[i+1] == u'{':
                r = check_bracket_pair(segs, i+1)
                if r == -1:
                    return False
            else:
                return False
            if i < l - 1 and segs[i+1] == u'{':
                r = check_bracket_pair(segs, i+1)
                if r == -1:
                    return False
            else:
                return False
        i += 1
    return True

def simplify_segs(segs):
    re_segs = []
    for s in segs:
        if s == u'@<other>' or s == u'<other>' or s == u'{划掉}':
            if len(re_segs) == 0 or re_segs[-1] != u'@NONVALIDSEQ':
                re_segs.append(u'@NONVALIDSEQ')
        else:
            re_segs.append(s)
    return re_segs

def semcheck(insegs, gtsegs):
    re_insegs = simplify_segs(insegs)
    re_gtsegs = simplify_segs(gtsegs)
    if len(re_gtsegs) != len(re_insegs):
        return False
    else:
        for i in range(len(re_gtsegs)):
            if re_gtsegs[i] != re_insegs[i]:
                return False

def perm_semantic(done_data):
    for sent in done_data:
        makeup_segs = done_data[sent][u'src']
        ed = round(len(makeup_segs) * 0.447)
        if random.random() < 0.178:
            ed = 0
        err_segs = list(makeup_segs)
        while editdistance.eval(makeup_segs, err_segs) < ed:
            etype = err_type[random.randint(0, len(err_type) - 1)]
            if etype == u'del':
                rnd_pos = random.randint(0, len(err_segs) - 1)
                if (err_segs[rnd_pos] not in available_words):
                    continue
                err_segs.pop(rnd_pos)
            elif etype == u'ins':
                rnd_pos = random.randint(0, len(err_segs))
                if (rnd_pos == 0 and err_segs[0] == u'@NONVALIDSEQ') or (rnd_pos > 0 and rnd_pos < len(err_segs) and (err_segs[rnd_pos-1] == u'@NONVALIDSEQ' or err_segs[rnd_pos] == u'@NONVALIDSEQ')) or (rnd_pos == len(err_segs) and err_segs[len(err_segs)-1] == u'@NONVALIDSEQ'):
                    rnd_e_tag = available_words[random.randint(0, len(available_words)-2)]
                else:
                    rnd_e_tag = available_words[random.randint(0, len(available_words)-1)]
                err_segs.insert(rnd_pos, rnd_e_tag)
            elif etype == u'rpl':
                rnd_pos = random.randint(0, len(err_segs) - 1)
                if (err_segs[rnd_pos] not in available_words):
                    continue
                if (len(err_segs) > 1 and rnd_pos == 0 and err_segs[1] == u'@NONVALIDSEQ') or (len(err_segs) > 2 and rnd_pos > 0 and rnd_pos < len(err_segs) - 1 and (err_segs[rnd_pos-1] == u'@NONVALIDSEQ' or err_segs[rnd_pos + 1] == u'@NONVALIDSEQ')) or (len(err_segs) > 1 and rnd_pos == len(err_segs) - 1 and err_segs[len(err_segs)-2] == u'@NONVALIDSEQ'):
                    rnd_e_tag = available_words[random.randint(0, len(available_words)-2)]
                else:
                    rnd_e_tag = available_words[random.randint(0, len(available_words)-1)]
                err_segs[rnd_pos] = rnd_e_tag
        done_data[sent][u'trg'] = err_segs
        print ' '.join(done_data[sent][u'src'])
        print ' '.join(done_data[sent][u'trg'])
        print '-------------------------------'

def perm_syntax(done_data):
    for sent in done_data:
        makeup_segs = done_data[sent][u'src']
        rnd = random.randint(0, 19)
        ed_num = 1
        if rnd <= 1:
            ed_num = 3
        elif rnd <= 3:
            ed_num = 2
        ed = 0
        err_segs = list(makeup_segs)
        while ed < ed_num:
            etype = u'dup'  # duplicate
            rnd = random.randint(0, 23)
            if rnd <= 1:
                etype = u'rep'  # replace
            elif rnd <= 5:
                etype = u'ins'  # insert
            elif rnd <= 9:
                etype = u'par'  # pair missing
            elif rnd <= 14:
                etype = u'mis'  # bracket missing

            if etype == u'dup':
                bars = []
                for i in range(len(err_segs)):
                    if err_segs[i] in (u'{', u'}'):
                        bars.append(i)
                if len(bars) == 0:
                    continue
                rnd = random.randint(0, len(bars) - 1)
                pos = bars[rnd]
                err_segs.insert(pos, err_segs[pos])
                ed += 1
            elif etype == u'rep':
                bars = []
                for i in range(len(err_segs)):
                    if err_segs[i] in (u'{', u'}'):
                        bars.append(i)
                if len(bars) == 0:
                    continue
                rnd = random.randint(0, len(bars) - 1)
                pos = bars[rnd]
                if err_segs[pos] == u'{':
                    err_segs[pos] = random.choice([u'(', u'['])
                elif err_segs[pos] == u'}':
                    err_segs[pos] = random.choice([u')', u']'])
                ed += 1
            elif etype == u'ins':
                rnd_pos = random.randint(0, len(err_segs))
                err_segs.insert(rnd_pos, random.choice([u'{', u'}']))
                ed += 1
            elif etype == u'par':
                bars = []
                for i in range(len(err_segs)):
                    if err_segs[i] in (u'{', u'}'):
                        bars.append(i)
                if len(bars) == 0 or len(bars) % 2 != 0:
                    continue
                rnd = random.randint(0, (len(bars) / 2) - 1) * 2
                pos_left = bars[rnd]
                pos_right = bars[rnd + 1]
                err_segs = err_segs[:pos_left] + err_segs[pos_right + 1:]
                ed += 1
            elif etype == u'mis':
                bars = []
                for i in range(len(err_segs)):
                    if err_segs[i] in (u'{', u'}'):
                        bars.append(i)
                if len(bars) == 0:
                    continue
                rnd = random.randint(0, len(bars) - 1)
                err_segs.pop(rnd)
                ed += 1
        done_data[sent][u'trg'] = err_segs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--syntax', help='If set, then generate syntactic permutation data, otherwise semantic.', action='store_true')
    opt = parser.parse_args()
    permtype = u'semantic'
    if opt.syntax:
        permtype = u'syntactic'

    # Load vocabulary
    with codecs.open(file_vocab) as f:
        for line in f:
            l = line.split()
            tag = l[0].decode('utf-8')
            vocab[tag] = int(l[1])
            for i in range(len(tag)):
                settrie.add(tag[:i+1])
            if int(l[2]) not in group2word:
                group2word[int(l[2])] = []
            group2word[int(l[2])].append(tag)
            word2group[tag] = int(l[2])
            if int(l[2]) > 0:
                available_words.append(tag)

    # Load given training data
    with codecs.open(os.path.join('data', file_train)) as f:
        load_dict = json.load(f)
        for key in load_dict:
            segs = segmentation(load_dict[key])
            data[key] = data[key] = {u'label' : u'correct', u'insegs' : segs}

    # Load given testing data inputs
    with codecs.open(os.path.join('data', file_test_in)) as f:
        load_dict = json.load(f)
        for key in load_dict:
            segs = segmentation(load_dict[key])
            test_data[key] = {u'label' : u'correct', u'insegs' : segs}

    # Insert @<other> into the vocabulary
    vocab[u"@<other>"] = 0

    # Check testing data input syntax
    '''
        with codecs.open(os.path.join('data', file_test_in)) as f:
            load_dict = json.load(f)
            for key in load_dict:
                segs = test_data[key][u'insegs']
                r = syncheck(segs)
                if r == False:
                    test_data[key][u'label'] = u'syntax_error'
    '''
    for key in test_data:
        segs = test_data[key][u'insegs']
        r = syncheck(segs)
        if r == False:
            test_data[key][u'label'] = u'syntax_error'

    # Check testing data ground truth syntax
    with codecs.open(os.path.join('data', file_test_gt)) as f:
        load_dict = json.load(f)
        for key in load_dict:
            segs = segmentation(load_dict[key])
            test_data[key][u'gtsegs'] = segs
            r = syncheck(segs)
            if r == False:
                test_data[key][u'label'] = u'gt_error'

    # Check testing data semantic
    for key in test_data:
        if test_data[key][u'label'] == u'correct':
            insegs = test_data[key][u'insegs']
            gtsegs = test_data[key][u'gtsegs']
            if semcheck(insegs, gtsegs) == False:
                test_data[key][u'label'] = u'semantic_error'

    # Simplify testing data segmentations (reduce invalid segments) if the ground truth is correct
    for key in test_data:
        if test_data[key][u'label'] != u'gt_error':
            insegs = test_data[key][u'insegs']
            gtsegs = test_data[key][u'gtsegs']
            test_data[key][u'src'] = simplify_segs(gtsegs)
            test_data[key][u'trg'] = simplify_segs(insegs)

    # Check traning data syntax and simplify segmentations if the syntax is correct
    '''
        with codecs.open(os.path.join('data', file_train)) as f:
            load_dict = json.load(f)
            for key in load_dict:
                segs = data[key][u'insegs']
                r = syncheck(segs)
                if r == False:
                    data[key][u'label'] = u'syntax_error'
                else:
                    data[key][u'simpsegs'] = simplify_segs(data[key][u'insegs'])
    '''
    for key in data:
        segs = data[key][u'insegs']
        r = syncheck(segs)
        if r == False:
            data[key][u'label'] = u'syntax_error'
        else:
            data[key][u'simpsegs'] = simplify_segs(data[key][u'insegs'])

    # Insert @NONVALIDSEQ into vocabulary
    group2word[1] = [u'@NONVALIDSEQ']
    word2group[u'@NONVALIDSEQ'] = 1
    available_words.append(u'@NONVALIDSEQ')

    # Remove duplicated training data 
    for key in data:
        if data[key][u'label'] == u'correct' and len(data[key][u'insegs']) > 0 and u''.join(data[key][u'simpsegs']) not in rare_data:
            rare_data[''.join(data[key][u'simpsegs'])] = {u'key':key, u'src':data[key][u'simpsegs'], u'trg':[]}

    print len(rare_data)

    # Generate training data src
    ss = 0
    while len(done_data) < data_size:
        rnd_id = random.randint(0, len(rare_data) - 1)
        sel_key = rare_data.keys()[rnd_id]
        src_segs = rare_data[sel_key][u'src']
        poslist = range(len(src_segs))
        random.shuffle(poslist)
        change_num = random.randint(0, len(src_segs) - 1)
        poslist = poslist[:change_num]
        makeup_segs = list(src_segs)
        for pos in poslist:
            sel_tag = src_segs[pos]
            sel_group_id = word2group[sel_tag]
            if sel_group_id == -1:
                continue
            sel_group = group2word[sel_group_id]
            rnd_voc = random.randint(0, len(sel_group) - 1)
            sel_perm = sel_group[rnd_voc]
            makeup_segs[pos] = sel_perm
        makeup_sent = u''.join(makeup_segs)
        if makeup_sent not in done_data and makeup_sent not in rare_data:
            ss += 1
            if ss % 5000 == 0:
                print ss
            done_data[makeup_sent] = {u'src':makeup_segs, u'trg':[]}
    
    # Add original data
    for sent in rare_data:
        done_data[sent] = {u'src':rare_data[sent][u'src'], u'trg':[]}

    prefix = u''
    # If semantic
    if permtype == u'semantic':
        prefix = u'se'
        perm_semantic(done_data)
    elif permtype == u'syntactic':
        prefix = u'sy'
        perm_syntax(done_data)
        new_test_data = {}
        for key in test_data:
            if test_data[key][u'label'] == u'syntax_error':
                new_test_data[key] = test_data[key]
        test_data = dict(new_test_data)

    available_words.append(u'{')
    available_words.append(u'}')

    with codecs.open(prefix + '-task2-train_src', 'w', 'utf-8') as train_src:
        with codecs.open(prefix + '-task2-train_trg', 'w', 'utf-8') as train_trg:
            with codecs.open(prefix + '-task2-dev_src', 'w', 'utf-8') as dev_src:
                with codecs.open(prefix + '-task2-dev_trg', 'w', 'utf-8') as dev_trg:
                    for sent in done_data:
                        trg = done_data[sent][u'src']
                        src = done_data[sent][u'trg']
                        r = random.random()
                        if r < dev_p:
                            dev_src.write(' '.join(src) + '\n')
                            dev_trg.write(' '.join(trg) + '\n')
                        else:
                            train_src.write(' '.join(src) + '\n')
                            train_trg.write(' '.join(trg) + '\n')
    with codecs.open(prefix + '-task2-test_src', 'w', 'utf-8') as test_src:
        with codecs.open(prefix + '-task2-test_trg', 'w', 'utf-8') as test_trg:
            for sent in test_data:
                if test_data[sent][u'label'] != u'gt_error':
                    trg = test_data[sent][u'src']
                    src = test_data[sent][u'trg']
                    test_src.write(' '.join(src) + '\n')
                    test_trg.write(' '.join(trg) + '\n')

    new_vocab = {}
    idx = 3
    for tag in available_words:
        new_vocab[tag] = idx
        idx += 1
    with open(prefix + '_src_voc_task2.pkl', 'w') as f:
        pickle.dump(new_vocab, f)
    with open(prefix + '_trg_voc_task2.pkl', 'w') as f:
        pickle.dump(new_vocab, f)
    with open(prefix + '_number_chars_task2', 'w') as f:
        pickle.dump(idx, f)
        pickle.dump(idx, f)
    print idx
    
