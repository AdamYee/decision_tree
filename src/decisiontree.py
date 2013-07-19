# Adam Yee
# COMP 251
# Project 1, Decision Trees. ID3 Algorithm

# Algorithm???:
# calculate P for examples (p1, p2, .. pn)
# calculate I(P)
# calculate gain for each feature
# chose HIG for feature, create a node R for that feature
# for v in values
#     add value link
#     subset examples = examples with value v for F
#     recurse, attach resulting subtree under link
# return subtree rooted at R


from collections import Counter
import copy
import math
import random
import sys

print("COMP 251 Project 1, Decision Trees")

def getExamples(filename):
    """
    param filename: filename of dataset to use
    
    returns examples: shuffled list of each record in the dataset
    """
    cars = '../datasets/car-evaluation/car.data'
    balance = '../datasets/balance-scale/balance-scale.data'
    examples = []
    if filename == 'car.data':
        f = open(cars)
    elif filename == 'balance-scale.data':
        f = open(balance)
    for line in f:
        line = line.rstrip().split(',')
        examples.append(line)
    random.shuffle(examples)
#    print(examples)
    return examples

def mapFeaturesToValues(examples, feature_names, category_pos):
    """
    Takes in a list of lists as examples, maps the feature names 
    to there values but excludes the category feature (and its values).
    """
    d = {}
    minusone = False
    for pos in range(len(feature_names)+1):
        if pos == category_pos:
            minusone = True
            continue
        l = []
        for record in examples:
            
            l.append(record[pos])
        values = set(l)
#        print(values)
        if minusone: pos -= 1
        d[feature_names[pos]] = list(values)

#    print(d)
    return d

def splitByTargetCategory(examples, outputs, category_position):
    """
    returns targets: a dictionary of target attribute value keywords 
        whose data is lists from the dataset
    """
    targets = {}
    for o in outputs:
        targets[o] = [] # dictionary setup
    for record in examples:
#        category = record.pop(category_position)
        category = record[category_position]
        targets.get(category).append(record)
    return targets

def splitByPercentage(examples, lesser):
    test_set = []
    starting_len = len(examples)
    for i in range(int(lesser*starting_len)):
#        test_set.append(examples.pop(random.randrange(len(examples))))
        test_set.append(examples.pop(i))
#    print(len(examples)/starting_len)
#    print(len(test_set)/starting_len)
    return (examples, test_set)


def entropy(examples):
    """
    param examples: dictionary of category features mapped
        to their data.
    
    returns entropy: Entropy of the dataset or subset
    """
    total=0
    ps = [] # probabilities list
    for l in examples.values():
        ps.append(len(l))
    for p in ps:
        total += p
    
    return entropy2(ps,total) 

def entropy2(ps, total):
#    print(ps,total)
    entropy = 0
    for p in ps: 
        if p: # if p is not zero
            p /= total
            entropy += -p*math.log(p,2)
    return entropy

def informationGain(S, feature, F_pos, mapped_features):
    """
    param S: dataset or subset
    param F_pos: feature position in each data entry line
    param feature: name of the feature
    mapped_features: dictionary mapping of feature to feature_values
    
    returns (gain, feature): the information gain for splitting S on F
    
    Notes: Gain = E(S) - SUM[for all values] (Sv/S)E(Sv)
    """    

    e = entropy(S)

    gain = e
    total = 0 # dataset or subset total    
    total_values = []
    
#    print("feature pos",F)
    for k,v in S.items():
        values = []
        for entry in v:
            values.append(entry[F_pos])
        # step 1
        total += len(values)
        total_values.append(Counter(values))

    # step 2
    # calculate E(Sv), entropy for each value of F
    fvalues = mapped_features[feature]
#    print(fvalues)
    for val in fvalues:
        # calculating weighted sum on all values of F
#        print(val)
        ps = [] # probabilites
        for c in total_values: # c is a Counter object
            if c.get(val):
                ps.append(c.get(val))
        subtotal = sum(ps)
#        print(ps,subtotal)
        if total: # to prevent zero division errors
            gain -= (subtotal/total) * entropy2(ps, subtotal)
        
#    print(gain)
    return (gain, feature)


#---------------------- Tree stuctures ------------------------#

def getTreeScore(tree, test_set, category_position):
    # Step one: makeOneDecision for each record in the tuning set.
    # Step two: for each decision, count how many answered correctly and wrong.
    # Step three: calculate the error (% of wrong answers)
    # Step four: scoreOfBestTree = 1 - error
    correct = 0
    wrong = 0
    for record in test_set:
#        print('debug1')
        ans = tree.makeOneDecision(record)
#        print('debug2')
        if ans == record[category_position]:
            correct += 1
        else:
            wrong += 1
    return 1 - (wrong / (correct+wrong))

class Node:
    def __init__(self, feature=None, values=None,
                 examples=None, remaining_features=None, answer=None, leaf=False):
        
        self.feature = feature
        self.values = values
        self.examples = examples
        self.remaining_features = remaining_features
        self.answer = answer
        self.links = []
        self.leaf = leaf
        self.depth = 0
        self.node_number = 0
        self.toBeDeleted = False
        self.num_below = 0
    
    def addLink(self, value_label):
        self.links.append(Link(value_label))
    
    def getLinkForValue(self, value):
        for link in self.links:
            if link.getVal() == value:
                return link
    
    def nPrint(self, d, num=0):
        self.depth = d
        if self.depth < 3:
            if self.leaf:
                print('\t'*self.depth+'leaf', self.answer)
            else:
                print('\t'*self.depth+'Node', self.feature, '(N='+str(self.node_number)+')', '(D='+str(self.toBeDeleted)+')')
                
                if self.depth >= 2:
                    num += 1 # number of interior nodes counter
                    for link in self.links:
                        # INSTEAD OF CALLING PRINT CALL A DIFFERENT FUNCTION TO COUNT NODES
                        self.num_below += link.lPrint(self.depth, num)
#                    print('\t'*self.depth+'# of interior nodes below this one,',self.num_below)
                    
                for link in self.links:
                    link.lPrint(self.depth)
                
        return self.depth # depth is incremented in the link print

    def nCount(self):
        if self.leaf:
            return [0,1] # 0 interior, 1 leaf
        else:
            sizes = [1,0]  
            for l in self.links:
                s = l.lCount()
                sizes[0] += s[0]
                sizes[1] += s[1]
            return sizes

    def askQuestion(self, record, dict_feature_index):
        if self.answer:
            return self.answer
        else:
            try:
                record_feature_value = record[dict_feature_index[self.feature]]
            except KeyError:
                sys.exit(1)
            for link in self.links:
                if link.getVal() == record_feature_value:
                    return link.passAlong(record, dict_feature_index)
                    continue
                
    def nodeNumberAll(self, tree):
        self.node_number = tree.start_num
        if not self.answer:
            tree.NPlusOne()
        if not self.leaf:
            for link in self.links:
                link.linkNumberAll(tree)
    
    def nodeMarkToDelete(self, node):
        if self.node_number == node and not self.leaf:
            self.toBeDeleted = True
        else:
            for link in self.links:
                link.linkMarkToDelete(node)
    
    def removeAndReplace_N(self):
        if self.toBeDeleted:
            self.leaf = True
            count = 0
#            try:
            for k,v in self.examples.items():
                l = len(v)
                if count < l:
                    count = l
                    self.answer = k
#            except AttributeError:
#                sys.exit(1)
            # now delete the children and their children and their...
            for link in self.links:
                link.deleteLink()
            self.links = None
        else:
            for link in self.links:
                link.removeAndReplace_L()
                    
    def deleteNode(self):            
        for link in self.links:
            link.deleteLink()
        self.links = None

class Link:
    def __init__(self, value_label):
        self.value_label = value_label
        self.subtree = None # this is just a Node
        
    def getVal(self):
        return self.value_label
    
    def lPrint(self, depth, num=0):
        return self.subtree.nPrint(depth+1, num)
    
    def lCount(self):
        return self.subtree.nCount()
    
    def passAlong(self, record, dict_feature_value):
        """
        Passing the record along for further questioning.
        """
        return self.subtree.askQuestion(record, dict_feature_value)
    
    def linkNumberAll(self, tree):
        self.subtree.nodeNumberAll(tree)
        
    def linkMarkToDelete(self, node):
        self.subtree.nodeMarkToDelete(node)
        
    def deleteLink(self):
        if self.subtree.leaf: # if the subtree Node is a leaf
            self.subtree = None # set leaf to None
        else:
            self.subtree.deleteNode()
            self.subtree = None
            
    def removeAndReplace_L(self):
        self.subtree.removeAndReplace_N()
        
class DTree:
    def __init__(self, dict_feature_index=None, dict_feature_values=None):
        self.root = None
        self.dict_feature_index = dict_feature_index
        self.dict_feature_values = dict_feature_values
        self.num_interior_nodes = 0
        self.num_leaf_nodes = 0
        self.test_set = []
        self.scoreOfBestTree = 0
        self.start_num = 1
        
    def tPrint(self):
        if self.root == None:
            print("Tree is Empty")
        else:
            self.root.nPrint(0) # start at depth reference 0
            
    def tCount(self):
        sizes = self.root.nCount()
        return sizes
    
    def makeOneDecision(self, record):
        """
        Passes in a single entry/record and returns the answer (category). 
        """
        return self.root.askQuestion(record, self.dict_feature_index)
    
    def numberNodes(self):
        """
        Number all of the interior nodes, 1 ... N.
        """
        self.start_num = 1
        self.root.nodeNumberAll(self)
    
    def NPlusOne(self):
        self.start_num += 1
    
    def markNodeToDelete(self, node):
        self.root.nodeMarkToDelete(node)
        
    def removeAndReplace(self):
        self.root.removeAndReplace_N()
        
    #---------------------- ID3 ------------------------#
    def id3(self, examples, features, split_style):
        """
        returns tree: the tree
        """
        # If all examples in one category, return leaf node with category label
        num_cats = 0
        answer = None
        for k,v in examples.items():
            num = len(v)
            if num: # counting number of categories with examples
                num_cats += 1
                answer = k    
                
        if num_cats == 1: # if only 1 category has all examples
            return Node(answer = answer, leaf = True)
        
        #Else if features = {}, return leaf node with most common category label
        elif not features:
            count = 0
            most_common = None
            for k,v in examples.items():
                num = len(v)
                if count < num: # determining most common category
                    count = num
                    most_common = k
            return Node(answer = most_common, leaf = True)
        
        else:
            # split_style = 1, Quinlan's info gain method
            # Else select feature F with highest information gain 
            # on examples and create a node R for it
            if split_style == 1:
                gains = []
                for f in features:
                    gains.append(informationGain(examples, f, self.dict_feature_index[f], self.dict_feature_values))
                chosen = max(gains)
                F = chosen[1]
            # split_style = 2, Random split method
            elif split_style == 2:
                F = random.choice(features)
            
            values = self.dict_feature_values[F]

            features.remove(F)
            rf = list(features)
            # (1) creating Node (1)
            node = Node(F, values, examples, features)

            # For each value vi of F
            for vi in values:                
                # (2) Add out-going link to node R labeled with the value vi (2)
                node.addLink(vi)
                
                # Let examplesi be subset of examples that have value vi for F
                subset = {}
                for cat,data in node.examples.items():
                    subset[cat] = []
                    for record in data:
                        rvalue = record[self.dict_feature_index[F]]
                        if rvalue == vi: # if record feature value is equal to link value
                            subset[cat].append(record)

                # (3) Recursively call DTree(examplesi, features - {F}) and 
                # attach resulting tree as subtree under Link (3)                
                node.getLinkForValue(vi).subtree = self.id3(subset, rf, split_style)
            
            # (4) Return subtree rooted at R (4)
            return node
             
# MAIN
def main():
    random.seed()
    pick = ''
    session = 0
    print('\nPress \'9\' to exit or end sessions during inputs')
    while pick != '9':
        print("\nAvailable data-sets:\n- car.data\t - balance-scale.data")
        print("Splitting method:\n- quinlan\t - random\t - prune")
        
        while not pick or pick not in ['car.data','balance-scale.data']:
            pick = input("Data-set choice > ")
            if pick == '9':
                print('exiting')
                return
            dataset = pick
        pick = None
        
        while not pick or pick not in ['quinlan','random','prune']:
            pick = input("Splitting choice > ")
            if pick == '9':
                print('exiting')
                return
            splitter = pick
        pick = None
            
        if dataset == 'car.data':
            outputs = ['unacc','acc','good','vgood']
            feature_names = ['buying','maint','doors','persons','lug_boot','safety']
            feature_index_mappings = {'buying':0,'maint':1,'doors':2,'persons':3,'lug_boot':4,'safety':5} # feature names mapped to list position
            category_position = 6
            
        elif dataset == 'balance-scale.data':
            outputs = ['L','B','R']
            feature_names = ['left-weight','left-distance','right-weight','right-distance']
            feature_index_mappings = {'left-weight':0,'left-distance':1,'right-weight':2,'right-distance':3}
            category_position = 0
            
        else:
            print('Not a valid choice')
            pick = None
            continue
        
        # make the D-Tree
        examples = getExamples(dataset)
#        examples = list(original_examples)
        mapped_features_to_values = mapFeaturesToValues(examples, feature_names, category_position)
        [training_set, test_set] = splitByPercentage(examples, 0.1) # split 90% 10%
        prune_training_set = []
        for record in training_set:
            prune_training_set.append(list(record))
#        print(id(training_set))
#        print(id(prune_training_set))
        training_set_dict = splitByTargetCategory(training_set, outputs, category_position)
#        test_set_dict = splitByTargetCategory(test_set, outputs, category_position)
        
        # Create the decision tree
        tree = None
        
        # Build the tree
        if splitter == 'quinlan':
            tree = DTree(feature_index_mappings, mapped_features_to_values)
            tree.root = tree.id3(training_set_dict, feature_names, 1)
            tree.test_set = test_set
            tree.numberNodes()
        elif splitter == 'random':
            tree = DTree(feature_index_mappings, mapped_features_to_values)
            tree.root = tree.id3(training_set_dict, feature_names, 2)
            tree.test_set = test_set
            tree.numberNodes()
        else: # PRUNE
            # (1) Given a training set (prune_training_set), create a grow set and a tuning set. (1)
            # (1) Place 20% of the examples in the tuning set and the remainder in the grow set. (1)
            tuning_set = []
            start_len = len(training_set)
            # Pop out 20% of the training_set into the tuning_set
            for i in range(int(0.2*start_len)):
                tuning_set.append( prune_training_set.pop (random.randrange (len(prune_training_set))))
            # Assign what's left of training set to grow set (80%)
            grow_set = prune_training_set
            grow_set_dict = splitByTargetCategory(grow_set, outputs, category_position)
            
            # (2) Create a tree that fully fits the grow set. Use the info_gain as the (2) 
            # (2) scoring function for features. Call this original tree, OrigTree.    (2)
            origTree = DTree(feature_index_mappings, mapped_features_to_values)
            origTree.root = origTree.id3(grow_set_dict, feature_names, 1)
            
            # (2) Initialize ScoreOfBestTree to the score of OrigTree on the examples in the tuning set. (2)
            sobt = getTreeScore(origTree, tuning_set, category_position)
#            print('sobt',sobt)
            
            # (3) Number all of the interior nodes, 1 ... N. (3)
            origTree.numberNodes()
#            print(origTree.start_num)
            # (4) Repeat the following 100 times: (4)
            K_times = 5
            tree = None
            bestScore = 0
            print('Pruning, please wait...')
            for n in range(500):
                print('...')
                # (4) Make a copy of OrigTree called CopiedTree. (4)
                copiedTree = copy.deepcopy(origTree)
                for k in range(K_times):
                    # (4) Uniformly pick a random number, D, between 1 and N. (4)
                    D = random.randrange(1,origTree.start_num)
                    # (4) Mark node D in CopiedTree as "to be deleted". (4)
                    copiedTree.markNodeToDelete(D)
                
                # (5) Traverse CopiedTree and remove all the nodes marked for deletion. (5)
                # (5) Replace the deleted node by the majority answer category for the subtree rooted at this node. (5)
                copiedTree.removeAndReplace()
                
                # (6) Score CopiedTree on the tuning set and keep track of the best tree generated (6)
                score = getTreeScore(copiedTree, tuning_set, category_position)
                if score > bestScore:
                    bestScore = score
                    tree = copiedTree # (7) Return the best tree found. (7)
                del(copiedTree)
            
            print('Pruning complete')
            tree.test_set = test_set
            tree.numberNodes()

        choice = ''
        session += 1
        print('\nSession',session)
        while choice != '9':
            print('\nOptions:\n1) dumpTree\t2) treeSize\t 3) Categorize test_set')
            while not choice:
                choice = input("choose one: ")
            if choice == '1':
                dumpTree(tree)
            elif choice == '2':
                treeSize(tree)
            elif choice == '3':
                score = getTreeScore(tree, tree.test_set, category_position)
                print('Error rate:',1-score)
            elif choice == '9':
                print('ending session')
                continue
            choice = ''
        
        del(tree)
        pick = None

def dumpTree(tree):
    tree.tPrint()
    
def treeSize(tree):
    s = tree.tCount()
    print('Interior:',s[0])
    print('Leaves:',s[1])
    print('Total:',sum(s))    

if __name__ == "__main__":
    main()
    

