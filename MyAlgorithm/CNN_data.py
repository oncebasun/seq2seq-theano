#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Produces data in the format:
# word(in letters) et_x 0 et_y 0 ... (file 1)
# and
# <number> (file 2)

# get_only_tree(w1, w2), applyOnlyTree(w1, eTree)
from getEditTrees import editTreesByPos, get_only_tree, applyOnlyTree
import io
from itertools import izip
import pickle as cPickle

def get_CNN_data (file_in, file_out, USE_CORPUS):
  
  if USE_CORPUS:
    print('\nERROR: USE_CORPUS is not implemented, yet.')
    exit()
    
  # Part 1: Find all edit trees in the training set.
  in_file = io.open(file_in + 'train', 'r', encoding='utf-8')
  voc_src = {}
  voc_trg = {}
  printToIndex = {}
  indexToTree = {}
  indexToFrequ = {}
  indexToTags = {} # all tags this et has been seen with

  for line in in_file:
    w1 = line.strip().split('\t')[0]
    w2 = line.strip().split('\t')[2]
    morph_tag = line.strip().split('\t')[1]
    
    for char in w1:
      if char not in voc_src:
	voc_src[char] = len(voc_src) + 3
 
    new_et = get_only_tree(w1, w2)
    
    if not new_et.myprint() in printToIndex:
      printToIndex[new_et.myprint()] = len(printToIndex)
      indexToTree[printToIndex[new_et.myprint()]] = new_et
      indexToFrequ[printToIndex[new_et.myprint()]] = 0
      indexToTags[printToIndex[new_et.myprint()]] = set()
    indexToFrequ[printToIndex[new_et.myprint()]] += 1
    indexToTags[printToIndex[new_et.myprint()]].add(morph_tag)
      
  for i in range(len(indexToTree)):
    voc_src['et' + str(i)] = len(voc_src) + 3
    voc_trg['et' + str(i)] = len(voc_trg) + 3
  voc_src['etUNK'] = len(voc_src) + 3
  voc_trg['etUNK'] = len(voc_trg) + 3
    
  in_file.close()
  
  # Part 2: Store the vocabulary files.
  outfile_src_voc = open(file_out+ '_src_voc.pkl', 'wb')
  outfile_trg_voc = open(file_out + '_trg_voc.pkl', 'wb')
  cPickle.dump(voc_src, outfile_src_voc)
  cPickle.dump(voc_trg, outfile_trg_voc)
  outfile_src_voc.close()
  outfile_trg_voc.close()
  
  print('Storing number characters')
  no_char_file = open(file_out + '_number_chars', 'wb')
  cPickle.dump(len(voc_src)+3, no_char_file)
  cPickle.dump(len(voc_trg)+3, no_char_file)
  no_char_file.close()
    
  print('vocabulary files done')
  print(len(voc_src)+3)
  print(len(voc_trg)+3)
    
  # Part 3: Make output files.
  for part in ['train', 'dev', 'test']: 
    output = {}
    
    if part == 'test' and not ('german' in file_in or 'arabic' in file_in):
      continue
    in_file = io.open(file_in + part, 'r', encoding='utf-8')
    
    for line in in_file:
      out_s = out_t = u''
      w1 = line.strip().split('\t')[0]
      w2 = line.strip().split('\t')[2]
      morph_tag = line.strip().split('\t')[1]
 
      new_et = get_only_tree(w1, w2) 
      
      if new_et.myprint() in printToIndex and indexToFrequ[printToIndex[new_et.myprint()]] > 1: # this has to be tested for dev and test set
        out_t = (u'et' + str(printToIndex[new_et.myprint()]) + '\n')
      else:
        out_t = (u'etUNK\n')
      out_s = (u' '.join(list(w1)))
      
      counter = 0
      for index, tree in indexToTree.iteritems():
	if indexToFrequ[index] <= 1 or morph_tag not in indexToTags[index]:
	  continue
	# Store every applicable tree.
        if not 'error' in applyOnlyTree(w1, tree):
	  counter += 1
	  # TODO: substitute this 0 by a one if the word appears in the corpus
	  if not USE_CORPUS:
	    out_s += (u' et' + str(index))
      out_s += (u'\n')
      #print(counter)	    
      output[out_s] = out_t
      
    out_src = io.open(file_out + '_' + part + '_src', 'w', encoding='utf-8')
    out_trg = io.open(file_out + '_' + part + '_trg', 'w', encoding='utf-8')
    for out_s, out_t in output.iteritems():
      out_src.write(out_s)
      out_trg.write(out_t)
    in_file.close()
    out_src.close()
    out_trg.close()
    print(part + ' done')
    #exit()	
      
  
  
  

if __name__ == "__main__":
  #LANG = 'arabic'
  languages = ['georgian', 'russian', 'turkish', 'navajo', 'spanish', 'finnish']
  USE_CORPUS = False
  
  for LANG in languages:
    file_in = '/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/' + LANG + '-task3-'
    file_out = 'TreeData/' + LANG + '-task3'
  
    get_CNN_data(file_in, file_out, USE_CORPUS)
  