# Import this in baseline_analysis.py to filter obviously wrong answers.

from getEditTrees import editTreesByPos
from getEditTrees import applyOnlyTree
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
  
  def __init__(self, lang):
    self.formToEt = editTreesByPos(lang)[2]
    
  def filterResult(self, lemma, form, word):
    allRight = False
    
    if lemma == word or form not in self.formToEt:
    #if form not in self.formToEt:
      return True
    
    for et in self.formToEt[form]:
      #print(lemma)
      #print(et)
      #print(applyOnlyTree(lemma, et))
      #print('****')
      if unicode(applyOnlyTree(lemma, et)) == word:
        allRight = True
    #sys.exit(0) 
     
    #if not allRight:
    #  print("filtered: " + lemma + ' : ' + form + ' : ' + word)
    return allRight
  
  def correctResult(self, lemma, form, word):
    # We cannot correct this.
    if form not in self.formToEt:
      return word
    
    # Choose the form that has the mistake at an edit tree border (and maybe for doubling)
    for et in self.formToEt[form]:
      newWord = unicode(applyOnlyTree(lemma, et))
      if edit_distance(newWord, word) == 1:
        return newWord # TODO: perform some choice here
      
    # No fitting solution has been found:
    return word
    
  def rightFormInEtSet(self, lemma, form, solution):
    hasSolution = False
    
    if form not in self.formToEt:
      return False
    
    for et in self.formToEt[form]:
      if unicode(applyOnlyTree(lemma, et)) == solution:
        hasSolution = True

    return hasSolution
