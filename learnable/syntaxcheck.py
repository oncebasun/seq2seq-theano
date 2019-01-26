# -*- coding: UTF-8 -*-
import os
import codecs
import json
import csv
import numpy as np
import editdistance


file_test_in = "autocorrection_sample_testing.json"
file_test_gt = "autocorrection_sample_testing_gt.json"
file_vocab = "vocab.txt"

vocab = {}
settrie = set()
data = {}

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

stat = []

def semcheck(insegs, gtsegs):
    re_insegs = simplify_segs(insegs)
    re_gtsegs = simplify_segs(gtsegs)
    ed = editdistance.eval(re_insegs, re_gtsegs)
    if len(re_gtsegs) != len(re_insegs):
        ped = float(ed) / len(re_gtsegs)
        stat.append(ped)
        return False
    else:
        for i in range(len(re_gtsegs)):
            if re_gtsegs[i] != re_insegs[i]:
                ped = float(ed) / len(re_gtsegs)
                stat.append(ped)
                return False
    
if __name__ == "__main__":

    with codecs.open(file_vocab) as f:
        for line in f:
            l = line.split()
            tag = l[0].decode('utf-8')
            vocab[tag] = int(l[1])
            for i in range(len(tag)):
                settrie.add(tag[:i+1])
    
    with codecs.open(os.path.join('data', file_test_in)) as f:
        load_dict = json.load(f)
        for key in load_dict:
            segs = segmentation(load_dict[key])
            data[key] = {u'label' : u'correct', u'insegs' : segs}

    vocab[u"@<other>"] = 0

    with codecs.open(os.path.join('data', file_test_in)) as f:
        load_dict = json.load(f)
        for key in load_dict:
            segs = data[key][u'insegs']
            r = syncheck(segs)
            if r == False:
                data[key][u'label'] = u'syntax_error'
                print load_dict[key]

    with codecs.open(os.path.join('data', file_test_gt)) as f:
        load_dict = json.load(f)
        for key in load_dict:
            segs = segmentation(load_dict[key])
            data[key][u'gtsegs'] = segs
            r = syncheck(segs)
            if r == False:
                data[key][u'label'] = u'gt_error'

    for key in data:
        if data[key][u'label'] == u'correct':
            insegs = data[key][u'insegs']
            gtsegs = data[key][u'gtsegs']
            if semcheck(insegs, gtsegs) == False:
                data[key][u'label'] = u'semantic_error'
    
    stat = np.array(stat)
    #print np.mean(stat)
    #print np.std(stat)

    count_label = {
        u'correct':0, 
        u'syntax_error':0, 
        u'gt_error':0, 
        u'semantic_error':0
    }
    for key in data:
        count_label[data[key][u'label']] += 1

    print count_label
