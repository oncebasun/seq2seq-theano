#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Usage: python getEditTrees.py [language]
#
# A tool to find all edit trees between words of the same lemma and store them by POS.

import codecs, sys, os 
from treefiles.capitalTree import CapitalTree
from treefiles.tree import Tree


# Finds the word that you get by applying edittree to w1.
# If the edit tree cannot be applied: returns "#error#".
def applyEditTree(w1, eTree):
  rememberUpper = w1[0].isupper()

  # Should convert an upper case word.
  if (rememberUpper and eTree.toUpperOrLowerCase == 1) or (not rememberUpper and eTree.toUpperOrLowerCase == -1):
    return "#error#"
  
  w1 = w1[0].lower() + w1[1:] 
  returnString = applyOnlyTree(w1, eTree.tree)

  if returnString == "#error#":
    return returnString
  
  if eTree.toUpperOrLowerCase == 0:
    if rememberUpper:
      return returnString[0].upper() + returnString[1:]
    else:
      return returnString
  if eTree.toUpperOrLowerCase == 1:
    return returnString[0].upper() + returnString[1:]
  if eTree.toUpperOrLowerCase == -1:
    return returnString
  

def applyOnlyTree(w1, eTree):
  # NOTE: Not testing if tree has children, but it should. So no error should occur here.
  if not eTree.data[0] == "sub":
    if len(w1) < (eTree.data[0] + eTree.data[1]):
      return "#error#"
    p = applyOnlyTree(w1[:eTree.data[0]], eTree.left)
    if p == "#error#":
      return "#error#"
    s = applyOnlyTree(w1[len(w1)-eTree.data[1]:], eTree.right)
    if s == "#error#":
      return "#error#"
    return p + w1[eTree.data[0]:len(w1) - eTree.data[1]] + s
  
  #From here on it is the "else", is to say, it is a substitution node.
  if w1 == eTree.data[1]:
    return eTree.data[2]
  return "#error#" 

def getLCS(w1, w2):
  os.environ["PYTHONHASHSEED"] = "0"
  switched = False
  if len(w2) > len(w1): 
    switched = True  # switched word order for faster calculation... keep this in mind for returning the result
    longest = w2
    shortest = w1
  else:
    longest = w1
    shortest = w2
  for i in range(len(shortest)):
    j = 0
    while j <= i:
      #print(shortest[i-j:len(shortest)-j])
      if shortest[i-j:len(shortest)-j] in longest:
          for k in range(len(longest)):
              if longest[k:k+(len(shortest)-j)-(i-j)] == shortest[i-j:len(shortest)-j]:
                  if not switched:
                    return shortest[i-j:len(shortest)-j], k, k+(len(shortest)-j)-(i-j), i-j, len(shortest)-j
                  else:
                    return shortest[i-j:len(shortest)-j], i-j, len(shortest)-j, k, k+(len(shortest)-j)-(i-j)
      j += 1
  return '', 0, 0, 0, 0


def get_only_tree(w1, w2):
  # Find start and end indices of the longest common substring.
  theLcs, i_s, i_e, j_s, j_e = getLCS(w1, w2)
  #print(w1 + '>' + str(i_s) + ':' + str(i_e) + ' ; ' + w2 + '>' + str(j_s) + ':' + str(j_e))
  #print('lcs: ' + theLcs)

  if i_e - i_s == 0:
    # Get substitution (node) here.
    returnTree = Tree(('sub', w1, w2))
    returnTree.left = None
    returnTree.right = None
    return returnTree
  
  returnTree = Tree((i_s, len(w1) - i_e))
  returnTree.left = get_only_tree(w1[:i_s], w2[:j_s])
  returnTree.right = get_only_tree(w1[i_e:], w2[j_e:])
  return returnTree

def find_edit_tree(w1, w2):
  capitalization = 0
  # Find first the right capitalization. 
  if (w1[0].islower() and w2[0].islower()) or (w1[0].isupper() and w2[0].isupper()):
    capitalization = 0
  if (w1[0].islower() and w2[0].isupper()):
    capitalization = 1
  if (w1[0].isupper() and w2[0].islower()):
    capitalization = -1
    
  w1 = w1[0].lower() + w1[1:]
  w2 = w2[0].lower() + w2[1:]
    
  return CapitalTree(get_only_tree(w1, w2), capitalization)


def editTreesByPos(lang):
  
  formToEt = {} # stores the possible eidt trees for each tag
  lemmaToEtAndTag = {}
  
  lemmataToForm = {} # storing all appearing forms (in tags) and words with each lemma, by POS

  lemmataToForm['V'] = {}
  lemmataToForm['ADJ'] = {}
  lemmataToForm['N'] = {}

  editTrees = {} # storing POS to edit tree

  # for German nouns, there should be 8 forms
  # however, the maximum we have is 4 (Geschlecht,Glasgeraet,Bruch,Fadenlauf)
  possiblePOS = ['V', 'ADJ', 'N']
  with codecs.open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/' + lang + '-task1-train', 'r', encoding="utf-8") as train_file:
    for line in train_file:
      parts = line.strip().split('\t')
      tags = parts[1].split(',')
      for tag in tags:
        if tag.split('=')[0] == 'pos':
          POS = tag.split('=')[1]
      if parts[0] not in lemmataToForm[POS]:
        lemmataToForm[POS][parts[0]] = [(parts[1], parts[2])]
      else:
        lemmataToForm[POS][parts[0]].append((parts[1], parts[2]))


  for aPos in possiblePOS:
    if aPos not in lemmataToForm:
      continue
    for lemma, forms in lemmataToForm[aPos].items():
      if lemma not in lemmaToEtAndTag: # should never happen to be; but better safe than sorry
        lemmaToEtAndTag[lemma] = set()
      
      for f1 in forms: # f1[0] is the tags, f1[1] is the word form
        if f1[1] == lemma:
          continue
        else: 
          et1 = get_only_tree(lemma, f1[1])
      
          # Small test:
          if f1[1] != applyOnlyTree(lemma, et1):
            print("error1")
            sys.exit(0)
           
          lemmaToEtAndTag[lemma].add((et1, f1[0]))
          
          # store tags to edit tree
          if f1[0] not in formToEt:
            formToEt[f1[0]] = set()
          formToEt[f1[0]].add(et1)

          if aPos not in editTrees:
            editTrees[aPos] = set()
          # here we should know if it is already in there
          editTrees[aPos].add(et1)

      '''
      for f1 in forms:
        for f2 in forms:
          if f1[1] == f2[1]:
            continue
          else:
            # TODO: calculate edit tree between 
            et1 = get_only_tree(f1[1], f2[1])
            et2 = get_only_tree(f2[1], f1[1])
          
            # Small tests:
            if f2[1] != applyOnlyTree(f1[1], et1):
              print("error1")
              sys.exit(0)
            if f1[1] != applyOnlyTree(f2[1], et2):
              print("error2")
              sys.exit(0)
            
            lemmaToEtAndTag[lemma].add((et1, f2[0]))

            if aPos not in editTrees:
              editTrees[aPos] = set()
            # here we should know if it is already in there
            editTrees[aPos].add(et1)
            editTrees[aPos].add(et2)
      '''
    #if aPos in editTrees:
    #  print(aPos + ':')
    #  print(len(editTrees[aPos]))

  return editTrees, lemmaToEtAndTag, formToEt

if __name__ == "__main__":
  editTreesByPos(sys.argv[1])