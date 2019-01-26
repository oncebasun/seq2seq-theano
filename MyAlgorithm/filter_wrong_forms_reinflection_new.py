#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import this in baseline_analysis.py to filter obviously wrong answers.

from getEditTrees_reinflection import editTreesByPos
from getEditTrees_reinflection import applyOnlyTree, verifyApplication
import sys


def edit_distance(s1, s2):
  m=len(s1)+1
  n=len(s2)+1

  tbl = {}
  for i in range(m): tbl[i,0]=i
  for j in range(n): tbl[0,j]=j
  for i in range(1, m):
    for j in range(1, n):
      cost = 0 if s1[i-1] == s2[j-1] else 1
      tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

  return tbl[i,j]

  
class Filter:
  
  def __init__(self, lang, resulting_predictions):
    self.formToEt = editTreesByPos(lang, resulting_predictions)[2]
    
  def filterResult(self, lemma, src_form, trg_form, word):
    allRight = False
    
    if lemma == word or (src_form, trg_form) not in self.formToEt:
    #if form not in self.formToEt:
      return True
    
    for et in self.formToEt[(src_form, trg_form)]:
      #print(lemma)
      #print(et)
      #print(applyOnlyTree(lemma, et))
      #print('****')
      if unicode(applyOnlyTree(lemma, et)) == word:
        allRight = True
      # Loose application, not THAT save, but at least not "verschlimmbessernd" ;):
      if verifyApplication(lemma, et, word):
	allRight = True
    #sys.exit(0) 
     
    #if not allRight:
    #  print("filtered: " + lemma + ' : ' + form + ' : ' + word)
    return allRight
  
  def correctResult(self, lemma, src_form, trg_form, word, corpus_voc = None):
    # We cannot correct this.
    if (src_form, trg_form) not in self.formToEt:
      return word
    
    # Choose the form that has the mistake at an edit tree border (and maybe for doubling)
    for et in self.formToEt[(src_form, trg_form)]:
      newWord = unicode(applyOnlyTree(lemma, et))
      if edit_distance(newWord, word) == 1:
	if not corpus_voc or newWord in corpus_voc:
          return newWord # TODO: perform some choice here
        #else:
	#  print('lalala')
      
    # No fitting solution has been found:
    return word
    
  def rightFormInEtSet(self, lemma, src_form, trg_form, solution):
    hasSolution = False
    
    if (src_form, trg_form) not in self.formToEt:
      return False
    
    for et in self.formToEt[(src_form, trg_form)]:
      if unicode(applyOnlyTree(lemma, et)) == solution:
        hasSolution = True

    return hasSolution
