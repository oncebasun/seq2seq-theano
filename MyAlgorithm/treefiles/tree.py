#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Tree(object):
  def __init__(self, data):
    self.left = None
    self.right = None
    self.data = data

  def myprint(self):
    if self==None:
      return u""
    else:
      mylen = len(self.data)
      if mylen==2:
        copystring = u"[copy"+str(self.data[0])+":"+str(self.data[1])+"]"
        leftn = self.left.myprint()
        rghtn = self.right.myprint()
        return leftn+copystring+rghtn
      elif mylen==3:
	#print(self.data[1])
	#print(self.data[2])
        substring = u'[sub:'+ self.data[1] +u'/'+ self.data[2] +u']'
        return substring
      else:
        assert 0
    
  def __eq__(self, otherTree):
    if not type(otherTree) is Tree:
      return False
    if self.data != otherTree.data:
      return False
    else:
      if self.left == None and otherTree.left != None:
        return False
      if self.left != None and otherTree.left == None:
        return False           
      if self.right == None and otherTree.right != None:
        return False
      if self.right != None and otherTree.right == None:
        return False           
            
      if self.right == None and otherTree.right == None and self.left == None and otherTree.left == None:
        return True
      return (self.right.__eq__(otherTree.right) and self.left.__eq__(otherTree.left))
    
  def  __hash__(self):
    if not (type(self.left) is Tree) and not (type(self.right) is Tree):
      return hash((hash("leftNone"), hash(self.data), hash("rightNone")))
    if not (type(self.left) is Tree):
      return hash((hash("leftNone"), hash(self.data), hash(self.right)))
    if not (type(self.right) is Tree):
      return hash((hash(self.left), hash(self.data), hash("rightNone")))
    return hash((hash(self.left), hash(self.data), hash(self.right)))
    
  
