#from tree import Tree
   
class CapitalTree(object):
  def __init__(self, tree, toUpperOrLowerCase):
    self.tree = tree
    self.toUpperOrLowerCase = toUpperOrLowerCase

  def myprint(self):
    if self==None:
      return ""
    else:
      if self.toUpperOrLowerCase == -1:
        returnString = "uppercase -> lowercase\n"
      if self.toUpperOrLowerCase == 1:
        returnString = "lowercase -> uppercase\n"
      if self.toUpperOrLowerCase == 0:
        returnString = "no change in capitalization\n"
      return returnString + self.tree.myprint()
    
  def __eq__(self, otherTree):
    if not type(otherTree) is CapitalTree:
      return False
    if self.toUpperOrLowerCase != otherTree.toUpperOrLowerCase:
      return False
    else:
      return self.tree.__eq__(otherTree.tree)
    
  def  __hash__(self):
    return hash((hash(self.toUpperOrLowerCase), hash(self.tree)))
    
  
