# Usage: testWordsInCorpus.py [language] {corpus file}
# If no corpus file is named, the programme will try to load a corresponding cPickle file.
#
# German corpus: /mounts/data/proj/huiming/SIGMORPHON/dewiki-20151102-pages-articles-multistream.xml
#
# This script finds words that should belong to a paradigm in the corpus and adds them (for training?).

from getEditTrees import editTreesByPos
from getEditTrees import applyOnlyTree
import sys
import pickle as cPickle


toAdd = {} # lemma to things that should be autocompleted
uniquenessCheck = {} # (lemma, form) -> word, avoiding that we add things we are unsure about
doneLemmas = set() # all lemmata that have already been looked at


# Just looking at the union.
# Returns a dictinary lemma -> (et, tags) with things to add to the original one.
def autoComplete(lemma1, etTag1, lemma2, etTag2, corpusWords):
  doneLemmas.add((lemma1, lemma2))
  doneLemmas.add((lemma2, lemma1))
  
  etAndTagToAdd2 = set()
  etAndTagToAdd1 = set()
  notFound = 0
  allRight1 = True
  allRight2 = True

  # Check for lemma 1:
  for (et, form) in etTag2.difference(etTag1):
    
    result = applyOnlyTree(lemma1, et)
    if result == '#error#':
      allRight1 = False
      break
    if result not in corpusWords or corpusWords[result] <=3:
      notFound += 1
      if notFound == 1:
        allRight1 = False
        break
    else:
      etAndTagToAdd1.add((et, form)) 
      
  # Check for lemma 2:
  for (et, form) in etTag1.difference(etTag2):
    
    result = applyOnlyTree(lemma2, et)
    if result == '#error#':
      allRight2 = False
      break
    if result not in corpusWords or corpusWords[result] <=3:
      notFound += 1
      if notFound == 1:
        allRight2 = False
        break
    else:
      etAndTagToAdd2.add((et, form)) 
 
  if allRight1 and allRight2:
    if etAndTagToAdd1:
      if lemma1 not in toAdd:
        toAdd[lemma1] = set()
      toAdd[lemma1] = toAdd[lemma1].union(etAndTagToAdd1)
      for (et, form) in etAndTagToAdd1:
        if (lemma1, form) not in uniquenessCheck:
          uniquenessCheck[(lemma1, form)] = set()
        uniquenessCheck[(lemma1, form)].add(applyOnlyTree(lemma1, et))
      
    if etAndTagToAdd2:
      if lemma2 not in toAdd:
        toAdd[lemma2] = set()
      toAdd[lemma2] = toAdd[lemma2].union(etAndTagToAdd2)
      for (et, form) in etAndTagToAdd2:
        if (lemma2, form) not in uniquenessCheck:
          uniquenessCheck[(lemma2, form)] = set()
        uniquenessCheck[(lemma2, form)].add(applyOnlyTree(lemma2, et))
        
        
# Lemma 1 has more ETs than lemma 2.
# Returns a dictinary lemma -> (et, tags) with things to add to the original one.
def autoComplete2(lemma1, etTag1, lemma2, etTag2, corpusWords):
  etAndTagToAdd = set()
  notFound = 0
  
  for (et, form) in etTag1.difference(etTag2):
    
    result = applyOnlyTree(lemma2, et)
    if result == '#error#':
      break
    if result not in corpusWords or corpusWords[result] <=3:
      notFound += 1
      if notFound == 2:
        break
    else:
      etAndTagToAdd.add((et, form)) 
 
  if etAndTagToAdd:
    if lemma2 not in toAdd:
      toAdd[lemma2] = set()
    toAdd[lemma2] = toAdd[lemma2].union(etAndTagToAdd)
    for (et, form) in etAndTagToAdd:
      if (lemma2, form) not in uniquenessCheck:
        uniquenessCheck[(lemma2, form)] = set()
      uniquenessCheck[(lemma2, form)].add(applyOnlyTree(lemma2, et))
  

# Test if a group of (edit tree, tag) combinations for a lemma is subset of the one for another lemma.
# If yes, try if the missing edit trees are applicable and if the corresponding word appears in the corpus.
def getAdditionalWords(lemmaToEtAndTag, corpusWords):
  isTrue = 0
  isFalse = 0
  for lemma1, etTag1 in lemmaToEtAndTag.items():
    for lemma2, etTag2 in lemmaToEtAndTag.items():
      if (lemma1, lemma2) in doneLemmas:
        continue
      if len(etTag1) <= 2 or len(etTag2) <= 2: # for now, don't complete things with 0 or only 1 entry. We are just not sure enough.
        isFalse += 1
        continue
      maybeSame = False
      theInter = etTag1.intersection(etTag2)
      if len(theInter) > 1 and (len(etTag1) > 2 + len(theInter) or len(etTag2) > 2 + len(theInter)): # TODO: use here the POS max
        maybeSame = True
        autoComplete(lemma1, etTag1, lemma2, etTag2, corpusWords)
        isTrue += 1
      else:
        isFalse += 1
         
  #print(str(len(toAdd)) +  ' words have been added.')
  #print("Is subset: " + str(isTrue))
  #print("No subset: " + str(isFalse))
  
  noWordsToAdd = 0
  for lemma, aSet in toAdd.items():
    noWordsToAdd += len(aSet)
  return noWordsToAdd

def announce(*objs):
    print("# ", *objs, file = sys.stderr)

if __name__ == "__main__":
  lang = sys.argv[1]
  if len(sys.argv) == 2:
    usePickle = True
  else:
    usePickle = False
    
  posToEt, lemmaToEtAndTag, formToEt = editTreesByPos(lang)
  
  for lemma, aSet in lemmaToEtAndTag.items():
    for (et, form) in aSet:
      if (lemma, form) not in uniquenessCheck:
        uniquenessCheck[(lemma, form)] = set()
      uniquenessCheck[(lemma, form)].add(applyOnlyTree(lemma, et))
      #print(applyOnlyTree(lemma, et))
  #sys.exit(0)

  if not usePickle:
    # Read the bonus corpus.
    announce('Start reading corpus...')
    corpusWords = {} # word to its frequency
    with open(sys.argv[2], 'r') as corpus_file:
      for line in corpus_file:
        #tokens = tokenize.word_tokenize(line.strip())
        tokens = line.strip().split(' ')
        for token in tokens:
          if token not in corpusWords:
            corpusWords[token] = 0
          corpusWords[token] += 1
    announce('Done reading corpus.')
    # Store the dictionary to a binary file.
    print('Store the dictionary with the corpus words to a binary file...')
    save_file = open('/mounts/data/proj/huiming/SIGMORPHON/corpusWords_' + lang, 'wb')
    cPickle.dump(corpusWords, save_file, -1)
    save_file.close()
    print('Done.')
  else:
    # Load the corpusWords dictionary.
    announce('Load the words with cPickle...')
    vocListFile = open('/mounts/data/proj/huiming/SIGMORPHON/corpusWords_' + lang, 'rb')
    corpusWords = cPickle.load(vocListFile)
    vocListFile.close()
    announce('Words loaded.')
   
  with open('../data/' + lang + '-task1-dev', 'r') as dev_file:
    dev_words = set()
    for line in dev_file:
      parts = line.strip().split('\t')
      dev_words.add((parts[0], parts[1]))
  
  lastNumber = 0
  noWordsToAdd = 1
  while noWordsToAdd > lastNumber:
    lastNumber = noWordsToAdd
    noWordsToAdd = getAdditionalWords(lemmaToEtAndTag, corpusWords)
    
    for lemma, aSet in lemmaToEtAndTag.items():
      if lemma in toAdd:
        lemmaToEtAndTag[lemma] = lemmaToEtAndTag[lemma].union(toAdd[lemma])
    announce('Number word to add: ' + str(noWordsToAdd))
   
  # The union did not work well for some reason. Therefore, use toAdd directly.
  additionalWordsCounter = 0
  newWordsInTestSet = 0
  with open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/' + lang + '-bigger-task1-train', 'w') as out_file: 
    with open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/' + lang + '-task1-train', 'r') as original_file:
      for line in original_file:
        out_file.write(line)
    for lemma, etAndTagSet in toAdd.items():
      for (et, form) in etAndTagSet:
        if len(uniquenessCheck[(lemma, form)]) > 1:
          continue
        out_file.write(lemma + '\t' + form + '\t' + applyOnlyTree(lemma, et) + '\n')
        if (lemma, form) in dev_words:
          newWordsInTestSet += 1
          print(lemma + '\t' + form + '\t' + applyOnlyTree(lemma, et))
        additionalWordsCounter += 1
        
  print(str(additionalWordsCounter) + ' words have been added.\n' + str(newWordsInTestSet) + ' of them are in the test set.')
      