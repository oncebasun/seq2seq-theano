# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

from MyAlgorithm.filter_wrong_forms_reinflection_new import Filter as Filter2 # the 'new' here is a test
from MyAlgorithm.filter_wrong_forms import Filter
import perceptron_c, align, codecs, sys, re, getopt

import pickle as cPickle
from itertools import izip
import sys
import datetime
import time
import io
import editdistance
from shutil import copyfile
from random import shuffle

import codecs
import argparse
import logging
import pprint
import sys
import os, signal

from RNN import configurations2 as configurations
from RNN.__init__BACKUP import main as rnnMain
#print('lala')
from RNN.__init__BACKUP import mainPredict as rnnPredict
from RNN.stream import get_tr_stream, get_test_stream, get_dev_stream
import RNN.sampling as sampling

###############################################################################
# IMPORTANT: NEVER TOUCH ANY OF THIS. ONLY CHANGES ALLOWED ARE VIA COMMAND LINE.

# The filter evaluating which results are right and which are wrong.
noFilter = True #TODO: change this via command line ONLY
# This is for the NN I used before. Don't activate this anymore; results were horrible.
loadModel = False 
# Defines which classifier should be used.
classifierToUse = 'rnn' # Options are: 'perceptron' (=baseline), 'rnn', 'nn' ('nn' should not be used anymore)
# Use trainRnn=True to train the RNN. For evaluation use trainRnn=False.
trainRnn = True # has to be set only when RNN is used TODO: change this via command line ONLY
# Defines if testing should be done with an ensemble. (0 means no, a number is the number of single networks)
use_ensemble = 1  # number of networks in the ensemble in testing; in training number of the currently trained network; TODO: change this via command line ONLY
# If the system should use corpus information for error correction.
# This has only effect when using it together with 'noFilter = False'.
use_corpus = False #TODO: change this via command line ONLY

######### command line flags
sample_from_prob = False
test_on_dev = False
the_way = 1 # how the answer will be produced
###############################################################################


class MorphModel:
    def __init__(self):
        self.features   = {'tolemma':None, 'fromlemma':None}
        self.classes    = {'tolemma':None, 'fromlemma':None}
        self.classifier = {'tolemma':None, 'fromlemma':None}
        
class Morph:

    def __init__(self):
        self.models = {}
        self.msdfeatures = None
        self.msdclasses = None
        self.msdclassifier = None        

    def generate(self, word, featurestring, mode):
        #print('Input: ' + word)
        """Generates an output string from an input word and target
            feature string. The 'mode' variable is either 'tolemma' or
            'fromlemma' """
        pos = re.match(r'pos=([^,]*)', featurestring).group(1)
        ins = ['<'] + list(word) + ['>']
        outs = []
        prevaction = 'None'
        position = 0
        while position < len(ins):            
            feats = list(train_get_surrounding_syms(ins, position, u'in_')) + \
               list(train_get_surrounding_syms(outs, position, u'out_', lookright = False)) + \
               ['prevaction='+prevaction] + [u'MSD:' + featurestring]
            feats = feature_pairs(feats)
            if usePerceptron:
              decision = self.models[pos].classifier[mode].decision_function(feats)
            else:
              decision = mlp.decision_function(self.models[pos].classifier[mode], feats)
            decision = sorted(decision, key = lambda x: x[1], reverse = True)
            prevaction = self._findmax(decision, prevaction, len(ins)-position-1)
            actionlength, outstring = interpret_action(prevaction, ins[position])
            outs.append(outstring)
            
            position += actionlength
        return ''.join(outs[1:-1])
            
    def _findmax(self, decision, lastaction, maxlength):
        """Find best action that doesn't conflict with last (can't del/ins/chg two in a row)
           and isn't too long (can't change/del more than what remains)."""
        #return decision # TODO: this is a hack. Rethink it
        if lastaction[0] == 'D' or lastaction[0] == 'C' or lastaction[0] == 'I':
            for x in xrange(len(decision)):
                if decision[x][0][0] != lastaction[0]:
                    if decision[x][0][0] == u'C' and len(decision[x][0][1:]) > maxlength:
                        continue
                    if decision[x][0][0] == u'D' and int(decision[x][0][1:]) > maxlength:
                        continue
                    return decision[x][0]
        else:
            return decision[0][0]
            
    def add_features(self, pos, features, classes, mode):
        """Adds a collection of feautures and classes to a pos model
           'mode' is either 'tolemma' or 'fromlemma'."""
        if pos not in self.models:
            self.models[pos] = MorphModel()
        self.models[pos].features[mode] = features
        self.models[pos].classes[mode] = classes
        
    def get_pos(self):
        """Simply lists all poses associated with a model."""
        return list(self.models.keys())

    def add_classifier(self, pos, classifier, mode):
        """Adds a classifier to a pos model in a certain mode."""
        self.models[pos].classifier[mode] = classifier
        
    def get_features(self, pos, mode):
        return self.models[pos].features[mode]

    def get_classes(self, pos, mode):
        return self.models[pos].classes[mode]

    def extract_task3(self, lang, path):
        
        # We use the msd/form combinations from all three
        msdform = set()
        lines = [line.strip() for line in codecs.open(path + lang +'-task1-train', "r", encoding="utf-8")]
        for l in lines:
            lemma, msd, form = l.split(u'\t')
            msdform.add((msd, form))
        lines = [line.strip() for line in codecs.open(path + lang +'-task2-train', "r", encoding="utf-8")]
        for l in lines:
            msd1, form1, msd2, form2 = l.split(u'\t')
            msdform.add((msd1, form1))
            msdform.add((msd2, form2))
        lines = [line.strip() for line in codecs.open(path + lang +'-task3-train', "r", encoding="utf-8")]
        for l in lines:
            form1, msd2, form2 = l.split(u'\t')
            msdform.add((msd2, form2))

        self.msdfeatures = []
        self.msdclasses = []
        for msd, form in msdform:
            formfeatures = extract_substrings(form)
            self.msdfeatures.append(formfeatures)
            self.msdclasses.append(msd)
                
    def extract_task1(self, filename, mode, path):
        """Parse a file and extract features/classes for
        mapping to and from a lemma form."""
    
        lemmas = {} # mapping from each lemma to all its possible forms (including the lemma itself)
        poses = set()
        lines = [line.strip() for line in codecs.open(path + filename, "r", encoding="utf-8")]
        for l in lines:
            if 'pos=' not in l:
                continue
            lemma, feats, form = l.split(u'\t')
            pos = re.match(r'pos=([^,]*)', feats).group(1)
            if lemma not in lemmas:
                lemmas[lemma] = []
                lemmas[lemma].append((lemma, 'pos=' + pos + ',lemma=true'))
            lemmas[lemma].append((form, feats)) # form is the word, feats are the tags
            if pos not in poses:
                poses.add(pos)

        pairs = []
        wordpairs = []
        for lemma in lemmas:
            lemmafeatures = lemmas[lemma]
            for x in lemmafeatures:
                for y in lemmafeatures:
                    if (x != y) and ('lemma=true' in x[1]) and (mode == 'fromlemma'):
                        pairs.append(tuple((x[0], y[0], y[1]))) # lemma, word, tags
                        wordpairs.append(tuple((x[0], y[0])))
                    elif (x != y) and ('lemma=true' in x[1]) and (mode == 'tolemma'):
                        pairs.append(tuple((y[0], x[0], y[1]))) # word, lemma, tags
                        wordpairs.append(tuple((y[0], x[0])))

        if ALIGNTYPE == 'mcmc':
            alignedpairs = mcmc_align(wordpairs, ALIGN_SYM)
        elif ALIGNTYPE == 'med':
            alignedpairs = med_align(wordpairs, ALIGN_SYM)
        else:
            alignedpairs = dumb_align(wordpairs, ALIGN_SYM)
        
        chunkedpairs = chunk(alignedpairs) # makes them basicall have the same length, I guess

        for pos in poses: # Do one model per POS
            features = []
            classes = []
            # sample pair: [(u'L', u'L'), (u'u', u'\xfc'), (u's', u's'), (u't', u't'), (u'_', u'e')]
            for idx, pair in enumerate(chunkedpairs):
                if 'pos=' + pos not in pairs[idx][2]:
                    continue
                instring = ['<'] + [x[0] for x in pair] + ['>']
                outstring = ['<'] + [x[1] for x in pair] + ['>']

                msdfeatures = [ pairs[idx][2] ] # don't separate features
                msdfeatures = ['MSD:' + f for f in msdfeatures] # just put MSD in front of them
                prevaction = 'None'
                for position in range(0, len(instring)): # len(instring) = len(outstring)! Because of chunking
                    thiscl, feats = train_get_features(instring, outstring, position)
                    classes.append(thiscl) # as a class, I should insert Umlaut
                    featurelist = list(feats) + msdfeatures + ['prevaction='+prevaction]
                    featurelist = feature_pairs(featurelist)
                    features.append(featurelist)
                    prevaction = thiscl
            self.add_features(pos, features, classes, mode)

def feature_pairs(f):
    """Expand features to include pairs of features 
    where one is always a f=v feature."""
    pairs = [x + ".x." + y for x in f for y in f if u'=' in y]
    return pairs + f
    
def dumb_align(wordpairs, align_symbol):
    alignedpairs = []
    for idx, pair in enumerate(wordpairs):
        ins = pair[0]
        outs = pair[1]
        if len(ins) > len(outs):
            outs = outs + align_symbol * (len(ins)-len(outs))
        elif len(outs) > len(ins):
            ins = ins + align_symbol * (len(outs)-len(ins))
            alignedpairs.append((ins, outs))
    return alignedpairs
    
def mcmc_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol = align_symbol)
    return a.alignedpairs
    
def med_align(wordpairs, align_symbol):
    a = align.Aligner(wordpairs, align_symbol = align_symbol, mode = 'med')
    return a.alignedpairs

def train_get_surrounding_syms(s, position, featureprefix, lookright = True):
    """Get surrounding symbols from a list of chunks and position.
    >>> s = ['<', u'a', u'b', u'u', u'_', u't', u'a', u'n', u'doka', '>']
    >>> train_get_surrounding_syms(s, 4, 'in_')
    set([u'nin_ta', u'nin_t', u'nin_tan', u'pin_u', u'pin_bu', u'pin_abu'])
    """
    leftfeats = set()
    rightfeats = set()
    if position == 0:
        leftfeats |= {u'p' + featureprefix + u'none'}
    if (position == len(s)) and lookright:
        rightfeats |= {u'n' + featureprefix + u'none'}
    if position > 0:
        left = ''.join(s[:position]).replace(u'_', u'')
        leftfeats |= {u'p' + featureprefix + left[x:] for x in [-1,-2,-3]}
    if (position < len(s)) and lookright:
        right = ''.join(s[position:]).replace(u'_', u'')
        rightfeats |= {u'n' + featureprefix + right[:x] for x in [1,2,3]}
    return leftfeats | rightfeats
    
def train_get_features(ins, outs, position):
    feats = set()
    # Get class first #
    if ins[position] == outs[position]:
        cl = "R"
    elif u'_' in ins[position]:
        cl = "I" + outs[position]
    elif u'_' in outs[position]:
        cl = "D" + unicode(len(ins[position]))
    else:
        cl = "C" + outs[position]
        
    # Get features of surrounding symbols #
    feats |= train_get_surrounding_syms(ins, position, u'in_')
    feats |= train_get_surrounding_syms(outs, position, u'out_', lookright = False)
    return cl, feats

def interpret_action(action, ins):
    """Interpret classifier class: return length of input to consume + output."""
    if action[0] == u'R':
        return (1, ins)
    elif action[0] == u'D':
        return int(action[1:]), u''
    elif action[0] == u'C':
        return len(action[1:]), action[1:]
    elif action[0] == u'I':
        return 0, action[1:]
    
def chopup(s, t):
    """Returns grouped alignment of two strings
       in such a way that consecutive del/ins/chg operations
       are grouped to be one single operation.
       The input is two 1-to-1 aligned strings where _ = empty string.
    >>> chopup(['ka__yyab','kaxx__xy'])
    (['k', 'a', u'_', 'yy', 'ab'], ['k', 'a', 'xx', u'_', 'xy'])
    """
    def action(inchar, outchar):
        if inchar == u'_':
            return 'ins'
        elif outchar == u'_':
            return 'del'
        elif inchar != outchar:
            return 'chg'
        else:
            return 'rep'
            
    idx = 1
    s = list(s)
    t = list(t)
    while idx < len(s):
        l = action(s[idx-1], t[idx-1])
        r = action(s[idx], t[idx])
        if (l == 'rep' and r == 'rep') or (l != r):
            s.insert(idx, ' ')
            t.insert(idx, ' ')
            idx += 1
        idx += 1
    s = tuple(u'_' if u'_' in x else x for x in ''.join(s).split(' '))
    t = tuple(u'_' if u'_' in x else x for x in ''.join(t).split(' '))
    return zip(s,t)
    
def chunk(pairs):
    """Chunk alignments to have possibly more than one symbol-one symbol."""
    chunkedpairs = []
    for instr, outstr in pairs:
        chunkedpairs.append(chopup(instr, outstr))
    return chunkedpairs
          
def extract_substrings(word):
    """Get len 2/3 substrings and return as list."""
    w3 = zip(word, word[1:], word[2:])
    w2 = zip(word, word[1:])
    return [''.join(x) for x in w2+w3]

def announce(*objs):
    print("***", *objs, file = sys.stderr)
    
def main(argv, resulting_predictions=None, the_track=None, bs_preds=None):
    global ALIGN_SYM
    global ALIGNTYPE
    global TASK
    
    ################################################  
      
    # This part is only used for the RNN if it is used.
    # Getting the command line arguments.

    # Get the language already, so we can pass it on to the saveto configuration.
    options, remainder = getopt.gnu_getopt(sys.argv[1:], 'l:t:a:d:e:tr:te:f:c:s:tr:emb:fa', ['language=','task=','align=', 'data=', 'ens=', 'train', 'test', 'filter', 'corpus', 'saveto=', 'track=', 'emb', 'finish_after='])
    PATH, ALIGN_SYM, ALIGNTYPE, TASK, DATAPATH, trainRnn, SAVETO, use_corpus, noFilter, TRACK, USE_EMBEDDINGS, finish_after = './', u'_', 'mcmc', 1, None, None, None, False, True, None, False, -1
    for opt, arg in options:
        if opt in ('-l', '--language'):
            LANGUAGE = arg
        if opt in ('-t', '--task'):
            TASK = int(arg)
        if opt in ('-d', '--data'):
            DATAPATH = arg
        if opt in ('-e', '--ens'): # in order to not have to change the code each time
            use_ensemble = int(arg)
        if opt in ('-tr', '--train'): # in order to not have to change the code each time
            #print('TRAIN')
            trainRnn = True
        if opt in ('-te', '--test'): # in order to not have to change the code each time
            #print('TEST')
            trainRnn = False
        if opt in ('-f', '--filter'): # in order to not have to change the code each time
            print('INFO: Filter used if testing.')
            noFilter = False
        if opt in ('-c', '--corpus'): # in order to not have to change the code each time
            print('INFO: Corpus used if testing.')
            use_corpus = True
        if opt in ('-s', '--saveto'):
            SAVETO = arg
        if opt in ('-tr', '--track'):
            TRACK = arg
        if opt in ('-emb', '--emb'):
            USE_EMBEDDINGS = True
        if opt in ('-fa', '--finish_after'):
            finish_after = int(arg)
      
    assert trainRnn != None 
    assert TRACK != None
      
    ################################################ 
   
    # My filter, filtering words with no corresponding edit tree.
    print(LANGUAGE)
    if noFilter:
      answerFilter = None
    else:
      if "-bigger" in LANGUAGE:
	if TASK == 1:
	  answerFilter = Filter(LANGUAGE[:len(LANGUAGE)-7])
	else:
	  answerFilter = Filter2(LANGUAGE[:len(LANGUAGE)-7], resulting_predictions)
      else:
	if TASK == 1 and not 'track2' in SAVETO:
	  answerFilter = Filter(LANGUAGE)
	else:
	  answerFilter = Filter2(LANGUAGE, resulting_predictions)
      
    if not useRnn:
      train = Morph()
      announce(LANGUAGE + ": learning alignment for form > lemma mapping")
      train.extract_task1(LANGUAGE + '-task1-train', 'fromlemma', PATH)
      if TASK == 2 or TASK == 3:
          announce(LANGUAGE + ": learning alignment for lemma > form mapping")
          train.extract_task1(LANGUAGE + '-task1-train', 'tolemma', PATH)

      if TASK == 1 or TASK == 2 or TASK == 3:
          for pos in train.get_pos():
              announce(LANGUAGE + ": training " + pos + " for lemma > form mapping")
              if usePerceptron:
                P = perceptron_c.Perceptron(shuffle = True, averaged = True, verbose = True, max_iter = 10, random_seed = 42)
                P.fit(train.get_features(pos, 'fromlemma'), train.get_classes(pos, 'fromlemma'))
                train.add_classifier(pos, P, 'fromlemma')
              elif loadModel:
                features, classes, feattoint, inttoclass, num_classes = mlp.prepare_data(train.get_features(pos, 'fromlemma'), train.get_classes(pos, 'fromlemma'))
                myMlp = mlp.MLP(feattoint, inttoclass, num_classes)
                myMlp = mlp.train_mlp((features, classes), myMlp, to_load = True, pos=LANGUAGE + '_' + pos,)
                train.add_classifier(pos, myMlp, 'fromlemma')
              else:
                features, classes, feattoint, inttoclass, num_classes = mlp.prepare_data(train.get_features(pos, 'fromlemma'), train.get_classes(pos, 'fromlemma'))
                myMlp = mlp.MLP(feattoint, inttoclass, num_classes)
                myMlp, best_params = mlp.train_mlp((features, classes), myMlp)
              
                # TO STORE: store parameter values:
                print('Saving net...')
                save_file = open('myMlp_2hidden_' + LANGUAGE + '_' + pos + '.p', 'wb')
                for p in best_params:
                  for p_part in p:
                    cPickle.dump(p_part, save_file, -1)
                save_file.close()
    
                #pickle.dump(myMlp, open( "mlp_" + pos + ".p", "wb" ) )   
                train.add_classifier(pos, myMlp, 'fromlemma')

      if TASK == 2 or TASK == 3:
          for pos in train.get_pos():
              announce(LANGUAGE + ": training " + pos + " for form > lemma mapping")
              P = perceptron_c.Perceptron(shuffle = True, averaged = True, verbose = True, max_iter = 10, random_seed = 42)
              P.fit(train.get_features(pos, 'tolemma'), train.get_classes(pos, 'tolemma'))
              train.add_classifier(pos, P, 'tolemma')

      if TASK == 3:
          if "-bigger" in LANGUAGE:
            train.extract_task3(LANGUAGE[:len(LANGUAGE)-7], PATH)
          else:
            train.extract_task3(LANGUAGE, PATH)
          announce(LANGUAGE + ": training form > msd classifier")
          train.msdclassifier = perceptron_c.Perceptron(shuffle = True, averaged = True, verbose = True, max_iter = 10, random_seed = 42)
          train.msdclassifier.fit(train.msdfeatures, train.msdclasses)
        
    myResults_file = io.open('/mounts/Users/cisintern/huiming/SIGMORPHON/Results/results_' + LANGUAGE, 'w', encoding='utf-8')
    if use_corpus:
      print('loading corpus data')
      corpus_data_file = open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/Corpora/corpus_voc_' + LANGUAGE + '_more_than_1.pkl', 'rb')
      corpus_voc = cPickle.load(corpus_data_file)
      corpus_data_file.close()
      print(len(corpus_voc))
    if 'paper' in SAVETO:
      middle = 'paper_version/'
    else:
      middle = ''
      
    if not os.path.exists('results/for_eval/' + middle + 'track2/'):
      os.makedirs('results/for_eval/' + middle + 'track2/')
    if noFilter:
       result_for_eval = io.open('results/for_eval/' + middle + 'track' + TRACK + '/' + LANGUAGE + '-task' + str(TASK) + '-solution', 'w', encoding='utf-8')
    else:
       result_for_eval = io.open('results/for_eval/ET_Filter/' + middle + 'track' + TRACK + '/' + LANGUAGE + '-task' + str(TASK) + '-solution', 'w', encoding='utf-8')
 
    if "-bigger" in LANGUAGE:
      testlines = [line.strip() for line in codecs.open(PATH+LANGUAGE[:len(LANGUAGE)-7] + '-task' + str(TASK) + '-dev', "r", encoding="utf-8")]
    else:
      #testlines = [line.strip() for line in codecs.open('data/data_given/' + LANGUAGE + '-task' + str(TASK) + '-train', "r", encoding="utf-8")]
      
      #testlines = [line.strip() for line in codecs.open('data/data_given/' + LANGUAGE + '-task' + str(TASK) + '-test-covered', "r", encoding="utf-8")] #TODO: USE THIS FOR REAL RESULTS!
      
      #testlines = [line.strip() for line in codecs.open('../data/created/' + LANGUAGE + '-task' + str(TASK) + '-test', "r", encoding="utf-8")] # this is for test, the original
      
      #testlines = [line.strip() for line in codecs.open('../data/created/' + LANGUAGE + '-task' + str(TASK) + '-test', "r", encoding="utf-8")] # this is for test
      
      # new
      testlines_src = [line.strip() for line in codecs.open(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-test_src', "r", encoding="utf-8")]
      testlines_trg = [line.strip() for line in codecs.open(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-test_trg', "r", encoding="utf-8")]
      
    # Make the new files.
    NEW_DATAPATH = DATAPATH.split(u'#')
    NEW_DATAPATH = NEW_DATAPATH[0] + '#' + str(int(NEW_DATAPATH[1]) + 1) + '#' + NEW_DATAPATH[2]
    if not os.path.exists(NEW_DATAPATH):
      os.makedirs(NEW_DATAPATH)

    copyfile(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-test_src_orig', NEW_DATAPATH + LANGUAGE + '-task1-test_src_orig')
    copyfile(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-test_src_orig', NEW_DATAPATH + LANGUAGE + '-task1-test_trg_orig')
    copyfile(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-test_src', NEW_DATAPATH + LANGUAGE + '-task1-test_src')
    copyfile(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-test_trg', NEW_DATAPATH + LANGUAGE + '-task1-test_trg')
    copyfile(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-dev_src', NEW_DATAPATH + LANGUAGE + '-task1-dev_src')
    copyfile(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-dev_trg', NEW_DATAPATH + LANGUAGE + '-task1-dev_trg')

    copyfile(DATAPATH + LANGUAGE + '_src_voc_task3.pkl', NEW_DATAPATH + LANGUAGE + '_src_voc_task3.pkl')
    copyfile(DATAPATH + LANGUAGE + '_trg_voc_task3.pkl', NEW_DATAPATH + LANGUAGE + '_trg_voc_task3.pkl')
    copyfile(DATAPATH + LANGUAGE + '_number_chars_task3', NEW_DATAPATH + LANGUAGE + '_number_chars_task3')

    train_lines = set()
    train_src = io.open(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-train_src', 'r', encoding='utf-8')
    train_trg = io.open(DATAPATH + LANGUAGE + '-task' + str(TASK) + '-train_trg', 'r', encoding='utf-8')
    train_new_src = io.open(NEW_DATAPATH + LANGUAGE + '-task' + str(TASK) + '-train_src', 'w', encoding='utf-8')
    train_new_trg = io.open(NEW_DATAPATH + LANGUAGE + '-task' + str(TASK) + '-train_trg', 'w', encoding='utf-8')
    for_train = []
    for l1, l2 in izip(train_src, train_trg):
      #train_new_src.write(l1)
      #train_new_trg.write(l2)
      train_lines.add((l1.strip(), l2.strip()))
      for_train.append((l1, l2))
    for w in bs_preds:
      lemma = u' '.join(list(w[1]))
      tag = u' '.join(w[0].split(u','))
      form = u' '.join(list(w[1]))
      if (tag + u' ' + lemma, form) not in train_lines:
        #train_new_src.write(tag + u' ' + lemma + '\n')
        #train_new_trg.write(form + '\n')
        for_train.append((tag + u' ' + lemma + '\n', form + '\n'))
    shuffle(for_train)
    for sth in for_train:
      train_new_src.write(sth[0])
      train_new_trg.write(sth[1])
    train_src.close()
    train_trg.close()
    train_new_src.close()
    train_new_trg.close()

      #testlines = [line.strip() for line in codecs.open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/' + LANGUAGE + '-task' + str(TASK) + '-dev', "r", encoding="utf-8")]
      #testlines = [line.strip() for line in codecs.open(configuration['src_data'], "r", encoding="utf-8")]
    if TASK == 1:  
        #print(resulting_predictions)
        #exit()
        all_correct_answers = set()
        specialTags = ['pos=V,tense=PRS,per=3,num=PL,aspect={IPFV/PFV}']
        # Initialize variables for error and filter analysis.
        errorCount = filterCount = correctFilter = rightCorrectionsCount = rightAnswersCount = totalCount = notFoundCount = specialRight = specialTotal = edit_distance = 0
        #for l in testlines:
        for (l_src, l_trg) in zip(testlines_src, testlines_trg): # this is the 'fake' test; is to say: part of the dev set
            totalCount += 1

            #split_result = l.split('\t')
            # using a test file WITH solutions:
            targetmsd = []
            lemma = []
            for sth in l_src.strip().split(u' '):
              if u'=' in sth:
                targetmsd.append(sth)
              else:
                lemma.append(sth)
            targetmsd = u','.join(targetmsd)
            lemma = u''.join(lemma)
            wordform = u''.join(l_trg.strip().split(u' '))
            '''
            if len(split_result) > 2:
              lemma, targetmsd, wordform = l.split('\t')
            else:
	      lemma, targetmsd = l.split('\t')
	      wordform = None
	      for_eval_file = u'\t'.join(l.split('\t'))
            '''

            if useRnn:
              if (targetmsd, lemma) not in resulting_predictions:
                myResults_file.write(lemma + '\t' + targetmsd + '\nNOT FOUND')
                #result_for_eval.write(for_eval_file + u'\t' + lemma + u'\n') # new emergency measure
                errorCount += 1
                notFoundCount += 1
                print('NOT FOUND:')
                print(u'' + targetmsd) 
                print(u'' + lemma)
                print()
                continue # should never happen when final prediction is made
              guess = resulting_predictions[(targetmsd, lemma)][0]
              guess_prob = resulting_predictions[(targetmsd, lemma)][1]
            else:
              guess = train.generate(lemma, targetmsd, 'fromlemma')
            if wordform != None:
	      if guess == wordform:
		myResults_file.write(lemma + '\t' + targetmsd + '\nRight: ' + wordform + '\tGuess:' + guess + ' (' + str(guess_prob) + ')\n')
                # TODO: add a right result here to the set
                all_correct_answers.add((targetmsd, lemma, wordform))
	      else:
		myResults_file.write(lemma + '\t' + targetmsd + '\nRight: ' + wordform + '\tGuess:' + guess + ' (' + str(guess_prob) + ')\t!!!\n')
            if not noFilter and not answerFilter.filterResult(lemma, targetmsd, guess):
              new_guess = answerFilter.correctResult(lemma, targetmsd, guess)
              if new_guess != guess:
                filterCount += 1
              guess = new_guess
            if targetmsd in specialTags:
	      specialTotal += 1
	      
	    if wordform != None:
              edit_distance +=  editdistance.eval(guess, targetform)
	      if u'' + guess != u'' + wordform: # The original version. The other one is just to see how much is filtered out.
		errorCount += 1
	      else: 
		if targetmsd in specialTags:
		  specialRight += 1
		rightAnswersCount += 1

	    if wordform == None:
	      result_for_eval.write(for_eval_file + u'\t' + guess + u'\n')

        if wordform != None:
	  print('Wrong answers: ' + str(errorCount) + '/' + str(len(testlines_src)) + '(' + str(notFoundCount) + ' not found)')
	  print('Right answers: ' + str(rightAnswersCount) + '/' + str(len(testlines_src)))
	  print('Accuracy: ' + str(rightAnswersCount*1.0/len(testlines_src)))
          print('Edit distance: ' + str(edit_distance*1.0/len(testlines_src)))
	  #print('Filtered: ' + str(filterCount))
	  #print('Right results for special tags: ' + str(specialRight) + '/' + str(specialTotal))
	else:
	  print('Finished. Check results in ' + 'results/for_eval/track' + TRACK + '/' + LANGUAGE + '_task' + str(TASK) + '-solution')
        myResults_file.close()

        bs_correct = 0
        for entry in bs_preds:
          if entry in all_correct_answers:
            bs_correct += 1
        print('Perc. of correctly bootstrapped samples: ' + str(1.0*bs_correct/len(bs_preds)))
        #print(all_correct_answers)
        exit()
 
    if TASK == 2:
        edit_distance = 0 # sum up edit distance and divide later
        # Initialize variables for error and filter analysis.
        errorCount = filterCount = wrongFilter = correctFilter = rightCorrectionsCount = rightAnswersCount = totalCount = notFoundCount = corpusFilter = corpusFilterWrong = corpusFilterRight = 0
        for (l_src, l_trg) in zip(testlines_src, testlines_trg):
            sourcemsd = l_src.split(u' ')[1].split(u'=')[1]
            sourceform = u''
            targetmsd = []
            for sth in l_src.split(u' '):
              if (u'LANG=' in sth or not u'=' in sth) and sth != u' ':
                sourceform = sourceform + sth
              elif u'=' in sth and not u'IN=' in sth:
                targetmsd.append(sth.split(u'=')[1])
            targetform = u''.join(l_trg.split(u' '))
            targetmsd = u','.join(targetmsd)
            #print(sourcemsd)
            #print(sourceform)
            #print(targetmsd)
            #print(targetform)
            #print(l_src)
            #print(l_trg)
            #exit()
	    #print(l)
	    #exit()
            for_eval_file = l_src + u'\t' + l_trg
            '''
	    split_result = l.split('\t')
	    if len(split_result) > 3: # should just be false
              sourcemsd, sourceform, targetmsd, targetform = l.split('\t')
              for_eval_file = u'\t'.join(l.split('\t')[:-1])
              #targetform = None
            else:
	      sourcemsd, targetmsd, sourceform = l.split('\t')# THE ORIGINAL was in other order
	      targetform = None
              for_eval_file = u'\t'.join(l.split('\t'))
            '''
            if useRnn:
              if (sourcemsd, targetmsd, sourceform) not in resulting_predictions:
                myResults_file.write(sourceform + '\t' + targetmsd + '\nNOT FOUND')
                result_for_eval.write(for_eval_file + u'\t' + sourceform + u'\n') # new emergency measure
                errorCount += 1
                notFoundCount += 1
                print('NOT FOUND:')
                print(u'' + targetmsd) 
                print(u'' + sourceform)
                print()
                continue # should never happen when final prediction is made
              guess = resulting_predictions[(sourcemsd, targetmsd, sourceform)][0] # you can get prob here!
              old_guess = guess
            else:
              lemma = train.generate(sourceform, sourcemsd, 'tolemma')
              guess = train.generate(lemma, targetmsd, 'fromlemma')
            if targetform != None:
              edit_distance +=  editdistance.eval(guess, targetform) # get edit distance between guess and target form and add it up
              if guess == targetform:
                myResults_file.write(sourceform + '\t' + targetmsd + '\nRight: ' + targetform + '\tGuess:' + guess + '\n')
              else:
                myResults_file.write(sourceform + '\t' + targetmsd + '\nRight: ' + targetform + '\tGuess:' + guess + '\t!!!\n')
            if not noFilter and (use_corpus and not guess in corpus_voc) and not answerFilter.filterResult(sourceform, sourcemsd, targetmsd, guess):
	      if use_corpus:
                new_guess = answerFilter.correctResult(sourceform, sourcemsd, targetmsd, guess, corpus_voc)
              else:
		new_guess = answerFilter.correctResult(sourceform, sourcemsd, targetmsd, guess)
              #if use_corpus and new_guess not in corpus_voc:
	#	new_guess = guess # don't change if we have never seen the new solution
              if new_guess != guess:
                filterCount += 1
                print(for_eval_file + u'\tOLD: ' + guess + u'\tNEW: ' + new_guess + '\n')
                exit()
              guess = new_guess
 
            if targetform != None:
              if u'' + guess != u'' + targetform:
	        if use_corpus and guess not in corpus_voc:
		  corpusFilterRight += 1
                errorCount += 1
                #out_file.write('Input: \t' + sourceform + ' : ' + sourcemsd + ' : ' + targetmsd + '\nRight sol: \t' + targetform + '\nOutput: \t' + guess + '\n\n')
                if u'' + old_guess == u'' + targetform:
		  wrongFilter += 1
              else: 
	        if use_corpus and guess not in corpus_voc:
		  corpusFilterWrong += 1
                rightAnswersCount += 1
                if u'' + old_guess != u'' + targetform:
		  correctFilter += 1
            if targetform == None:
	      result_for_eval.write(for_eval_file + u'\t' + guess + u'\n')
              
        if targetform != None:
	  print('Wrong answers: ' + str(errorCount) + '/' + str(len(testlines_src)) + '(' + str(notFoundCount) + ' not found)')
	  print('Right answers: ' + str(rightAnswersCount) + '/' + str(len(testlines_src)))
	  print('Accuracy: ' + str(rightAnswersCount*1.0/len(testlines_src)))
          print('Edit distance: ' + str(edit_distance*1.0/len(testlines_src)))
	  if not noFilter:
	    if use_corpus:
	      print('Filtered (using corpus): ' + str(filterCount))
	    else:
	      print('Filtered (without corpus): ' + str(filterCount))
	    print('Filter errors: ' + str(wrongFilter))
	    print('Corrected by filter: ' + str(correctFilter))
	  if use_corpus and noFilter:
	    print('Errors detected by corpus filter: ' + str(corpusFilterRight))
	    print('Right solutions erroneously filtered by corpus: ' + str(corpusFilterWrong))
	  #out_file.close()
	else:
	  print('Finished. Check results in ' + 'results/for_eval/track' + TRACK + '/' + LANGUAGE + '_task' + str(TASK) + '-solution')


    if TASK == 3:
        edit_distance = errorCount = filterCount = correctFilter = rightCorrectionsCount = rightAnswersCount = totalCount = notFoundCount = 0
        for (l_src, l_trg) in zip(testlines_src, testlines_trg):
            sourceform = u''
            targetmsd = []
            for sth in l_src.strip().split(u' '):
              if not u'=' in sth:
                sourceform = sourceform + sth
              else:
                if not u'LANG' in sth:
                  targetmsd.append(sth)
            targetform = u''.join(l_trg.strip().split(u' '))
            targetmsd = u','.join(targetmsd)

        #for l in testlines:
	#    split_result = l.split('\t')
	#    if len(split_result) > 2: 
        #      sourceform, targetmsd, targetform = l.split('\t')
        #    else:
	#      sourceform, targetmsd = l.split('\t')
	#      targetform = None
	#      for_eval_file = u'\t'.join(l.split('\t'))
            
            if useRnn:
              if (targetmsd, sourceform) not in resulting_predictions:
                myResults_file.write(sourceform + '\t' + targetmsd + '\nNOT FOUND')
                #result_for_eval.write(for_eval_file + u'\t' + sourceform + u'\n') # new emergency measure
                errorCount += 1
                notFoundCount += 1
                print('NOT FOUND:')
                print(u'' + targetmsd) 
                print(u'' + sourceform)
                print()
                continue # should never happen when final prediction is made
              guess = resulting_predictions[(targetmsd, sourceform)][0] # you can get prob here!
            else:
              sourcemsd = train.msdclassifier.predict(extract_substrings(sourceform))
              lemma = train.generate(sourceform, sourcemsd, 'tolemma')
              guess = train.generate(lemma, targetmsd, 'fromlemma')

            if targetform != None:
              edit_distance +=  editdistance.eval(guess, targetform) 
	      if u'' + guess != u'' + targetform:
		errorCount += 1
	      else: 
		rightAnswersCount += 1
            
            if targetform == None:
              result_for_eval.write(for_eval_file + u'\t' + guess + u'\n')
            
        if targetform != None:
	  print('Wrong answers: ' + str(errorCount) + '/' + str(len(testlines_src)) + '(' + str(notFoundCount) + ' not found)')
	  print('Right answers: ' + str(rightAnswersCount) + '/' + str(len(testlines_src)))
	  print('Accuracy: ' + str(rightAnswersCount*1.0/len(testlines_src)))
	  print('Edit distance: ' + str(edit_distance*1.0/len(testlines_src)))
          #print('Filtered: ' + str(filterCount))
	else:
	  print('Finished. Check results in ' + 'results/for_eval/track' + TRACK + '/' + LANGUAGE + '_task' + str(TASK) + '-solution')
            
     
# Prepares the RNN, but includes parsing of the path. TODO: check here if testing on dev or test
def prepareRnn(parser, lang, trainRnn, use_ensemble=0, data_path=None, SAVETO=None, the_task=None, track=None, finish_after=-1, load_config=True):
    print('Test_on_dev: ' + str(test_on_dev))
    logger = logging.getLogger(__name__)
    assert track != None
    # Get the arguments
    if use_ensemble == 1 or trainRnn or load_config:
      parser.add_argument("--proto",  default="get_config_cs2en",
                        help="Prototype config to use for config")
      parser.add_argument("--bokeh",  default=False, action="store_true",
                        help="Use bokeh server for plotting")
      
    args = parser.parse_known_args()[0]

    # Get configurations for model
    configuration = getattr(configurations, args.proto)()

    #print(configuration['finish_after'])
    #exit()
    
    configuration['the_task'] = the_task
    assert configuration['the_task'] == the_task 
      
    if configuration['allTagsSplit'] == 'allTagsSplit/':
      dataPath = '/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/allTagsSplit/'
      configuration['saveto'] += '_' + lang + '_' + str(configuration['enc_nhids']) + '_' + str(configuration['dec_nhids']) #+ '_allTagsSplit'

    elif configuration['allTagsSplit'] == 'POSextra/':
      dataPath = '/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/POSextra/'
      configuration['saveto'] += '_' + lang + '_' + str(configuration['enc_nhids']) + '_' + str(configuration['dec_nhids']) + '_POSextra'
      print('POS extra')
    else:
      dataPath = '/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/'
      configuration['saveto'] += '_' + lang + '_' + str(configuration['enc_nhids']) + '_' + str(configuration['dec_nhids'])
    
    # Use the path to data (if given):
    if data_path != None:
      dataPath = data_path
      
    if finish_after != -1:
      configuration['finish_after'] = finish_after
    print('Finish after ' + str(configuration['finish_after']) + ' epochs')
    #exit()
      
    # Modify the path to store the model according to the used parameters.
    if not configuration['use_attention']:
      configuration['saveto'] += '_noAttention'
    
    model_type_name = SAVETO
    configuration['saveto'] = model_type_name + 'task' + str(configuration['the_task']) + '/' + 'Ens_' + str(use_ensemble) + '_'  + configuration['saveto'] #TEST

    if trainRnn == False and 'track2' in configuration['saveto'] and 'task1' in configuration['saveto']:
      configuration['saveto'] = configuration['saveto'].split('task1')
      configuration['saveto'] = configuration['saveto'][0] + 'task2' + configuration['saveto'][1]
    
    if not os.path.exists(configuration['saveto']):
      os.makedirs(configuration['saveto'])

    if configuration['the_task'] > 1:
      no_char_file = open(dataPath + lang + '_number_chars_task' + str(configuration['the_task']), 'rb')
    else:
      no_char_file = open(dataPath + lang + '_number_chars', 'rb')
    configuration['src_vocab_size'] = cPickle.load(no_char_file)
    configuration['trg_vocab_size'] = cPickle.load(no_char_file)

    no_char_file.close()
    
    configuration['corpus_data'] = configuration['corpus_data'] + lang + '.pkl'
    
    if trainRnn:
      configuration['lang'] = lang # needed for the validation results file
      if data_path != None: # a special data path has been given
	configuration['src_data'] = data_path + lang + configuration['src_data'][1]
        configuration['trg_data'] = data_path + lang + configuration['trg_data'][1] 
      else:
        configuration['src_data'] = configuration['src_data'][0] + configuration['allTagsSplit'] + lang + configuration['src_data'][1]
        configuration['trg_data'] = configuration['trg_data'][0] + configuration['allTagsSplit'] + lang + configuration['trg_data'][1]  

      # Dev set exists for all languages now, but is called 'test' (for historical reasons).
      if data_path != None: # a special data path has been given
	configuration['val_set'] = data_path + lang + '-task' + str(configuration['the_task']) + '-dev_src' #data_path + lang + configuration['val_set'][1]
        configuration['val_set_grndtruth'] = data_path + lang + '-task' + str(configuration['the_task']) + '-dev_trg' #data_path + lang + configuration['val_set_grndtruth'][1]
        if data_path.split('/')[1] == 'data_exp_0504': # a special data path has been given
	  configuration['val_set'] = 'data/data_exp_0504/german-task1-dev_src' #data_path + lang + configuration['val_set'][1]
          configuration['val_set_grndtruth'] = 'data/data_exp_0504/german-task1-dev_trg' #data_path + lang + configuration['val_set_grndtruth'][1]
        if 'data_lemmatizer' in data_path: # a special data path has been given
	  configuration['val_set'] = 'data/data_lemmatizer/german-task1-dev_src' #data_path + lang + configuration['val_set'][1]
          configuration['val_set_grndtruth'] = 'data/data_lemmatizer/german-task1-dev_trg' #data_path + lang + configuration['val_set_grndtruth'][1]
      else:
        configuration['val_set'] = configuration['val_set'][0] + configuration['allTagsSplit'] + lang + configuration['val_set'][1]
        configuration['val_set_grndtruth'] = configuration['val_set_grndtruth'][0] + configuration['allTagsSplit'] + lang + configuration['val_set_grndtruth'][1]  
      copyConfig(configuration)
    else: # for testing we use the dev set, because this is what we hava for all languages
      if data_path != None:
        if test_on_dev:
          # a hack!
	  #configuration['src_data'] = data_path + lang + '-task' + str(configuration['the_task']) + '-dev_src'
	  #configuration['trg_data'] = data_path + lang + '-task' + str(configuration['the_task']) + '-dev_trg'
          configuration['src_data'] = '/mounts/Users/cisintern/huiming/universal-mri/Data/_ST_FINAL/optimal_dev/t2/'  + lang + '-task' + str(configuration['the_task']) + '-dev_src'
          configuration['trg_data'] = '/mounts/Users/cisintern/huiming/universal-mri/Data/_ST_FINAL/optimal_dev/t2/'  + lang + '-task' + str(configuration['the_task']) + '-dev_trg'
        else:
	  configuration['src_data'] = data_path + lang + '-task' + str(configuration['the_task']) + '-test_src'
	  configuration['trg_data'] = data_path + lang + '-task' + str(configuration['the_task']) + '-test_trg'
      else:

        configuration['src_data'] = configuration['src_data'][0] + configuration['allTagsSplit'] + lang + '-task' + str(configuration['the_task']) + '-dev_src'
      testlines = [line.strip() for line in codecs.open(configuration['src_data'], "r", encoding="utf-8")]
      configuration['batch_size'] = len(testlines) + 1
    if data_path != None:
      configuration['src_vocab'] = data_path + lang + configuration['src_vocab'][1]
      configuration['trg_vocab'] = data_path + lang + configuration['trg_vocab'][1]
    else:
      configuration['src_vocab'] = configuration['src_vocab'][0] + configuration['allTagsSplit'] + lang + configuration['src_vocab'][1]
      configuration['trg_vocab'] = configuration['trg_vocab'][0] + configuration['allTagsSplit'] + lang + configuration['trg_vocab'][1]
    
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
    
    configuration['src_data'] = configuration['src_data'].split('task')[0] + 'task' + str(configuration['the_task']) + configuration['src_data'].split('task')[1][1:]
    configuration['trg_data'] = configuration['trg_data'].split('task')[0] + 'task' + str(configuration['the_task']) + configuration['trg_data'].split('task')[1][1:]
    if configuration['the_task'] > 1:
      configuration['src_vocab'] = configuration['src_vocab'].split('.pkl')[0] + '_task' + str(configuration['the_task']) + '.pkl'
      configuration['trg_vocab'] = configuration['trg_vocab'].split('.pkl')[0] + '_task' + str(configuration['the_task']) + '.pkl'
    #if configuration['val_set'] != None:
    #  configuration['val_set'] = configuration['val_set'].split('task')[0] + 'task' + str(configuration['the_task']) + configuration['val_set'].split('task')[1][1:]
    #  configuration['val_set_grndtruth'] = configuration['val_set_grndtruth'].split('task')[0] + 'task' + str(configuration['the_task']) + configuration['val_set_grndtruth'].split('task')[1][1:]
      
    if trainRnn:
      #print(USE_EMBEDDINGS)
      #exit()
      configuration["rng_value"] *= use_ensemble
      rnnMain(configuration, get_tr_stream(**configuration),
           get_dev_stream(config=configuration), args.bokeh, the_track=track, the_task=the_task, use_embeddings=USE_EMBEDDINGS, lang=lang)
    else:
      rnnPredict(configuration, get_test_stream(**configuration), use_ensemble, lang, args.bokeh, the_track=track, sample_from_prob=sample_from_prob, the_way=the_way) 

        
# @pred_list: results of the single networks in a dictionary key -> (word, prob)
def ensemble_results_old(pred_list, method = 'adding_prob'):
  total_preds = {}
  all_ensemble_results = {}
  returned_preds = {} # only containing the best answer with a probability (when available)
  
  # adding_prob means that the single probs are added and then the highest one gets chosen
  if method == 'adding_prob':
    for resulting_predictions in pred_list:
        for key, value in resulting_predictions.iteritems():
            if key not in total_preds:
	      total_preds[key] = {value[0]: value[1]}
	      all_ensemble_results[key] = set()
	    else:
	      if value[0] not in total_preds[key]:
		total_preds[key][value[0]] = 0
	      total_preds[key][value[0]] += value[1]
	      all_ensemble_results[key].add(value[0])
	      
  # adding_appearances means that we choose the one which was selected most
  if method == 'adding_appearances':
    for resulting_predictions in pred_list:
        for key, value in resulting_predictions.iteritems():
            if key not in total_preds:
	      total_preds[key] = {value[0]: 1}
	      all_ensemble_results[key] = set()
	    else:
	      if value[0] not in total_preds[key]:
		total_preds[key][value[0]] = 0
	      total_preds[key][value[0]] += 1
	      all_ensemble_results[key].add(value[0])
	            
  # Now: get the maximum (=best answer).
  for key, value in total_preds.iteritems():
    for w in sorted(value, key=value.get, reverse=True):
      final_predictions[key] = (w, value[w]) # add only the result that is most frequent; TODO: change this to probability
      break
      
  return final_predictions, all_ensemble_results


def convert_format(inp, task, track=1):
  split_word = inp.split(' ')

  if task == 1 and track == 1:
    orig_form_array = []
    trg_tag_array = []
    for sth in split_word:
      if u'=' in sth:
	trg_tag_array.append(sth)
      else:
	orig_form_array.append(sth)

    return (u','.join(trg_tag_array), u''.join(orig_form_array))
  
  if task == 1 and track == 2:
    orig_form_array = []
    trg_tag_array = []
    for sth in split_word:
      if u'OUT=' in sth:
	trg_tag_array.append(sth.split(u'UT=')[1])
      else:
	if not u'IN=LEMMA' in sth:
	  orig_form_array.append(sth)

    return (u','.join(trg_tag_array), u''.join(orig_form_array))
  
  if task == 2:
    orig_form_array = []
    orig_tag_array = []
    trg_tag_array = []
    for sth in split_word:
      if 'IN=' in sth:
	orig_tag_array.append(sth.split('IN=')[1])
      elif 'OUT=' in sth:
	trg_tag_array.append(sth.split('OUT=')[1])
      else:
	orig_form_array.append(sth)
    return (u','.join(orig_tag_array), u','.join(trg_tag_array), u''.join(orig_form_array))
  
  if task == 3:
    orig_form_array = []
    trg_tag_array = []
    for sth in split_word:
      if u'=' in sth:
	trg_tag_array.append(sth)
      else:
	orig_form_array.append(sth)

    return (u','.join(trg_tag_array), u''.join(orig_form_array))
  
  
# @pred_list: results of the single networks in a dictionary key -> (word, prob)
# A new version, without probs.
def ensemble_results(pred_list, method = 'adding_prob', task=2, track=None):
  total_preds = {}
  all_ensemble_results = {}
  returned_preds = {} # only containing the best answer with a probability (when available)
  
  # Define the track for task 1, because of the format.
  track = 1
  if task == 1:
    for old_key, value in pred_list[0].iteritems():
      if u'IN=LEMMA' in old_key:
	track = 2
	continue
      
  # adding_prob means that the single probs are added and then the highest one gets chosen
  if method == 'adding_prob':
    for resulting_predictions in pred_list:
        for key, value in resulting_predictions.iteritems():
	    orig_form, orig_tag, trg_tag = convert_format(key, task)
            if key not in total_preds:
	      total_preds[key] = {value[0]: value[1]}
	      all_ensemble_results[key] = set()
	    else:
	      if value[0] not in total_preds[key]:
		total_preds[key][value[0]] = 0
	      total_preds[key][value[0]] += value[1]
	      all_ensemble_results[key].add(value[0])
	      
  # adding_appearances means that we choose the one which was selected most
  if method == 'adding_appearances':
    for resulting_predictions in pred_list:
        for old_key, value in resulting_predictions.iteritems():
	    #orig_form, orig_tag, trg_tag = convert_format(key, task)
	    key = convert_format(old_key, task, track)
	    #print(key)
            if key not in total_preds:
	      total_preds[key] = {value: 1}
	      all_ensemble_results[key] = set()
	    else:
	      if value not in total_preds[key]:
		total_preds[key][value] = 0
	      total_preds[key][value] += 1
	      all_ensemble_results[key].add(value)
	            
  # Now: get the maximum (=best answer).
  for key, value in total_preds.iteritems():
    for w in sorted(value, key=value.get, reverse=True):
      final_predictions[key] = (w, value[w]) # add only the result that is most frequent
      break
      
  return final_predictions, all_ensemble_results

def copyConfig(config):
  outfile = open(config['saveto']+ '/config', 'a')
  outfile.write('\n' + str(time.time()) + '\n')
  
  for k, v in config.iteritems():
    outfile.write(k + '\t' + str(v) + '\n')
  
  
if __name__ == "__main__":

    # Needed, so I can redirect the output.
    sys.stdout=codecs.getwriter('utf-8')(sys.stdout)
    
    print('Sample usage:')
    print('python2 ./baseline.py --task=2 --language=german --data=data/switched_order/ --train --saveto=final_models_ST/ [--ens=1 --corpus --filter]\nOR\npython2 ./baseline.py --task=2 --language=german --data=data/final_data_ST_track2/ --train --saveto=final_models_ST_track2/ --ens=1 &> logs/german_track2_task2_1 &')
    
    ###############################################
    # This is necessary, because I have 3 options.#
    ###############################################
    if classifierToUse == 'perceptron':
      usePerceptron = True
      trainRnn = False
      useRnn = False
    elif classifierToUse == 'nn':
      usePerceptron = False
      trainRnn = False
      useRnn = False
    elif classifierToUse == 'rnn':
      usePerceptron = False
      trainRnn = trainRnn # should be set above
      useRnn = True
    else:
      print('ERROR: No valid value for \'classifierToUse\' variable given.')
      sys.exit(0)
    ################################################  
      
    # This part is only used for the RNN if it is used.
    # Getting the command line arguments.
    if useRnn:
      # Get the language already, so we can pass it on to the saveto configuration.
      options, remainder = getopt.gnu_getopt(sys.argv[1:], 'l:t:a:d:e:tr:te:f:c:s:tr:sfp:tod:w:emb:fa:altconf', ['language=','task=','align=', 'data=', 'ens=', 'train', 'test', 'filter', 'corpus', 'saveto=', 'track=', 'sample_from_prob', 'test_on_dev', 'ways=', 'emb', 'finish_after=', 'alt_config'])
      PATH, ALIGN_SYM, ALIGNTYPE, TASK, DATAPATH, trainRnn, SAVETO, TRACK, USE_EMBEDDINGS, finish_after, ALT_CONF = './', u'_', 'mcmc', 1, None, None, None, None, False, -1, False
      for opt, arg in options:
          if opt in ('-l', '--language'):
              LANGUAGE = arg
          if opt in ('-t', '--task'):
              TASK = int(arg)
          if opt in ('-d', '--data'):
              DATAPATH = arg
          if opt in ('-e', '--ens'): # in order to not have to change the code each time
              use_ensemble = int(arg)
          if opt in ('-tr', '--train'): # in order to not have to change the code each time
              #print('TRAIN')
              trainRnn = True
          if opt in ('-te', '--test'): # in order to not have to change the code each time
              #print('TEST')
              trainRnn = False
          if opt in ('-f', '--filter'): # in order to not have to change the code each time
              print('INFO: Filter used if testing.')
              noFilter = False
          if opt in ('-c', '--corpus'): # in order to not have to change the code each time
              print('INFO: Corpus used if testing.')
              use_corpus = True
          if opt in ('-s', '--saveto'):
              SAVETO = arg
          if opt in ('-tr', '--track'):
              TRACK = arg
          if opt in ('-sfp', '--sample_from_prob'):
              sample_from_prob = True
          if opt in ('-tod', '--test_on_dev'):
              test_on_dev = True
          if opt in ('-w', '--ways'):
              the_way = arg
          if opt in ('-emb', '--emb'):
              USE_EMBEDDINGS = True
          if opt in ('-fa', '--finish_after'):
              finish_after = int(arg)
          if opt in ('-altconf', '--alt_config'):
              from RNN import alternative_configurations as configurations
              
      assert (DATAPATH != None or not trainRnn) and trainRnn != None and (SAVETO != None or not trainRnn) # change of policy: data path actually needed
      assert TRACK != None
      ################################################ 
      
      parser = argparse.ArgumentParser()   
      
      #print(LANGUAGE)
      #exit()
      
      #TASK=2
      #print(TASK)
      #exit()
      # get the results of track 2 to another place
      if False:
      #if 'paper' in SAVETO:
	middle = 'paper_version/'
      else: 
	#middle = ''
	middle ='__' + '__'.join(SAVETO.split('/')) + '/'

      the_new_path = 'results/' + middle + 'track' + TRACK + '/task' + str(TASK) + '/' + LANGUAGE

      if trainRnn:
        prepareRnn(parser, LANGUAGE, trainRnn, use_ensemble, data_path=DATAPATH, SAVETO=SAVETO, the_task=TASK, track=TRACK, finish_after=finish_after)
      else:
	if not os.path.exists(the_new_path+'/'):
        #if True:
          os.makedirs(the_new_path)
          
        # this was originally inside the loop
          for i in range(use_ensemble, use_ensemble+1):
            prepareRnn(parser, LANGUAGE, trainRnn, i, data_path=DATAPATH, SAVETO=SAVETO, the_task=TASK, track=TRACK)
            exit()
        else:
	  print('\nThere are already results for this language at ' + the_new_path + '. Using those...\n')

    # For predictions:
    final_predictions = {}
    collected_predictions = []
    all_ensemble_results = {}
    if not trainRnn and useRnn:
      for i in range(use_ensemble, use_ensemble+1):

        results_file = open(the_new_path + '/Ens_' + str(i) + '_intermediate_results.pkl', 'rb')
        res_accuracy = cPickle.load(results_file)
        res_ed = cPickle.load(results_file)
        results_file.close()
        print('Accuracy: ' + str(res_accuracy))
        print('Edit distance: ' + str(res_ed))
        exit()
        resulting_predictions = cPickle.load(open(the_new_path + '/Ens_' + str(i) + '_intermediate_results.pkl', 'rb'))
	collected_predictions.append(resulting_predictions)

      final_predictions, all_ensemble_results = ensemble_results(collected_predictions, 'adding_appearances', TASK, track=TRACK)
      #print(final_predictions)
      #print(len(final_predictions))
      #exit()

      # Load the bootstrapping predictions.
      bootstrap_preds = cPickle.load(open(SAVETO + 'task' + str(TASK) + '/Ens_' + str(use_ensemble) + '_model_' + LANGUAGE + '_100_100/test_sol.pkl', 'rb'))
      # Get the percentage of predictions to be added to train from the dev score (use half of it).
      dev_res_file = open(SAVETO + 'task' + str(TASK) + '/Ens_' + str(use_ensemble) + '_model_' + LANGUAGE + '_100_100/validation/accuracies_' + LANGUAGE, 'r') # warning: hard coded
      for line in dev_res_file:
        if len(line.strip().split(u'\t')) > 1:
          dev_accuracy = float(line.strip().split(u'\t')[1])

      # Step 1): Load a corpus to see if a word exists.
      if '-' in LANGUAGE:
        corpus = cPickle.load(io.open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/Corpora/corpus_voc_' + LANGUAGE.split('-')[0] + '.pkl', 'rb'))
      else:
        corpus = cPickle.load(io.open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/Corpora/corpus_voc_' + LANGUAGE + '.pkl', 'rb'))

      # Step 2): get the form-tag ratio for each morphological tag.
      form_lemma_ratio = {}
      train_src = io.open(DATAPATH + LANGUAGE + '-task3-train_src', 'r', encoding='utf-8')
      train_trg = io.open(DATAPATH + LANGUAGE + '-task3-train_trg', 'r', encoding='utf-8') 
      for l1, l2 in izip(train_src, train_trg):
        tag = []
        lemma = []
        form = u''.join(l2.strip().split(u' '))
        for sth in l1.strip().split(u' '):
          if u'=' in sth:
            tag.append(sth)
          else:
            lemma.append(sth)
        tag = u','.join(tag)
        lemma = u''.join(lemma)
        if form in corpus and lemma in corpus:
          if tag in form_lemma_ratio:
            form_lemma_ratio[tag] = (form_lemma_ratio[tag] + corpus[form] * 1.0 / corpus[lemma]) / 2
          else:
            form_lemma_ratio[tag] = corpus[form] * 1.0 / corpus[lemma]
      #print(form_lemma_ratio)
      #exit()    
 
      # Sort by probability (certainty of the model) and take the top best depending on dev accuracy.    
      counter = 0  
      bootstrapped_set = set()
      for w in sorted(bootstrap_preds, key=bootstrap_preds.get, reverse=True):
        #if bootstrap_preds[w] < 0.995:
        #if counter == 2:
        if counter >= dev_accuracy * 1.75 * len(bootstrap_preds): # Look only at the half that is in the dev accuracy.
          break
        counter += 1
        tag = []
        lemma = []
        for sth in w[0].strip().split(u' '):
          if u'=' in sth:
            tag.append(sth)
          else:
            lemma.append(sth)
        tag = u','.join(tag)
        lemma = u''.join(lemma)
        form = u''.join(list(w[1]))
        #print(w)
        #print(bootstrap_preds[w])
        # and (dev_accuracy > 0.5 or form != lemma)
        if form in corpus and corpus[form] >= 10 and (tag in form_lemma_ratio and lemma in corpus and corpus[form]*1.0/corpus[lemma] > form_lemma_ratio[tag] - 0.01*form_lemma_ratio[tag] and corpus[form]*1.0/corpus[lemma] < form_lemma_ratio[tag] + 0.01*form_lemma_ratio[tag]):
          bootstrapped_set.add((tag, lemma, form))
    
      #print(bootstrapped_set)
      print('Number of samples in the highest (half of dev acc.): ' + str(counter))
      print('Number of samples used for bootstrapping: ' + str(len(bootstrapped_set)))
      #exit()

      main(sys.argv, final_predictions, the_track=TRACK, bs_preds=bootstrapped_set)
    if usePerceptron:
      main(sys.argv)

    print('INFO: Finished')
