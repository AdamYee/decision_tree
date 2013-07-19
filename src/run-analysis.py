'''
Created on Sep 25, 2011

@author: adam
'''

from decisiontree import *

# (1) Partition data set into 10 parts (1)
def chunks(lst, n):
    """ Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def partition(dataset, K_size):
    chunksize = int(math.ceil(len(dataset)/float(K_size)))
    partitioned = []
    for k in chunks(dataset, chunksize):
        partitioned.append(k)
    return partitioned

def makeChunks(filename):
    """ Partitions dataset into K chunks. 
    """
    dataset = getExamples(filename) # shuffled dataset
    return partition(dataset, 10)

# (2) Cross validate to obtain average error rate of partitions (2)

def kfoldCrossValidation(filename, method, training_set, test_set):
    """ Returns the average
    error rate of all chunks.
    """
    
    if filename == 'car.data':
        outputs = ['unacc','acc','good','vgood']
        feature_names = ['buying','maint','doors','persons','lug_boot','safety']
        feature_index_mappings = {'buying':0,'maint':1,'doors':2,'persons':3,'lug_boot':4,'safety':5} # feature names mapped to list position
        category_position = 6
        
    elif filename == 'balance-scale.data':
        outputs = ['L','B','R']
        feature_names = ['left-weight','left-distance','right-weight','right-distance']
        feature_index_mappings = {'left-weight':0,'left-distance':1,'right-weight':2,'right-distance':3}
        category_position = 0
    
    mapped_features_to_values = mapFeaturesToValues(training_set, feature_names, category_position) 
    
    training_set_dict = splitByTargetCategory(training_set, outputs, category_position)
            
    # Now calculate error
    if method == 'prune':
        pass
    else:
        tree = DTree(feature_index_mappings, mapped_features_to_values)
        tree.test_set = test_set #################<------
            
        if method == 'quinlan':
            tree.root = tree.id3(training_set_dict, feature_names, 1) # type 1 = quinlan
                                 
        elif method == 'random':
            tree.root = tree.id3(training_set_dict, feature_names, 2) # type 2 = random                                      

        error = 1 - getTreeScore(tree, tree.test_set, category_position)
        (p,m) = confidence(error, len(tree.test_set))
#        print(error, p, m)

    return error, p, m

# (3) Apply 95% confidence to error rates
def confidence(error, n):
    plus = error + 1.96 * math.sqrt( error * (1 - error) / n )
    minus = error - 1.96 * math.sqrt( error * (1 - error) / n )
    return (plus, minus) 

def generalization(filename, method):
    chunks = makeChunks(filename)
    err_ave_list = []
    pave_list = [] # plus/positive
    mave_list = [] # minus/negative
    K = 10
    for fold in range(K):
        print(fold),
        validation_set = chunks[fold]
        training_set = []
        for chunk in chunks[:fold] + chunks[fold+1:]:
            training_set += chunk
        (err, p, m) = kfoldCrossValidation(filename, method, training_set, validation_set)
        err_ave_list.append(err)
        pave_list.append(p)
        mave_list.append(m)
    err_ave = 1/float(K)*sum(err_ave_list)
    pave = 1/float(K)*sum(pave_list) # plus ave
    mave = 1/float(K)*sum(mave_list) # minus ave
    if method == 'random':
        return (err_ave, pave, mave)
    else:
        print('(err,p,m):',err_ave, pave, mave)
    
def analyzeTreeSize(filename):
#    Vary the value of K in your pruning algorithm between 1 and N-1, where 
#    N is the total number of nodes in your original tree. Draw a single figure
#    whose x-axis is the size of the tree learned on the training data and whose 
#    y-axis is the error rate of the tree on the testing data. Discuss your 
#    experimental results and whether or not they uphold Occam's Razor.
    if filename == 'car.data':
        outputs = ['unacc','acc','good','vgood']
        feature_names = ['buying','maint','doors','persons','lug_boot','safety']
        feature_index_mappings = {'buying':0,'maint':1,'doors':2,'persons':3,'lug_boot':4,'safety':5} # feature names mapped to list position
        category_position = 6
    elif filename == 'balance-scale.data':
        outputs = ['L','B','R']
        feature_names = ['left-weight','left-distance','right-weight','right-distance']
        feature_index_mappings = {'left-weight':0,'left-distance':1,'right-weight':2,'right-distance':3}
        category_position = 0
        
    # create the tree
    dataset = getExamples(filename)
    random.shuffle(dataset)
    (training_set, test_set) = splitByPercentage(dataset, 0.1)
    mapped_features_to_values = mapFeaturesToValues(dataset, feature_names, category_position)
    
    # (1) Given a training set (prune_training_set), create a grow set and a tuning set. (1)
    # (1) Place 20% of the examples in the tuning set and the remainder in the grow set. (1)
    prune_training_set = copy.deepcopy(training_set)
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
    
    # number the nodes
    origTree.numberNodes()
    N = origTree.start_num - 1
    
    error_rates = []
    
    # vary K between 1 and N-1
    for num in range(1,N):
        # calculate error
        
        # (4) Repeat the following 100 times: (4)
        K_times = num
        bestScore = 0
        print('Pruning',num,'node(s), please wait...')
        for n in range(100):
#            print('...')
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
            
            s = copiedTree.tCount()
            tree_size = sum(s)     
            
            del(copiedTree)
        
        error = 1 - bestScore
        error_rates.append((tree_size, error))
        
    return error_rates

def analyzeROC(filename, method):
    if filename == 'car.data':
        outputs = ['unacc','acc','good','vgood']
        feature_names = ['buying','maint','doors','persons','lug_boot','safety']
        feature_index_mappings = {'buying':0,'maint':1,'doors':2,'persons':3,'lug_boot':4,'safety':5} # feature names mapped to list position
        category_position = 6
    elif filename == 'balance-scale.data':
        outputs = ['L','B','R']
        feature_names = ['left-weight','left-distance','right-weight','right-distance']
        feature_index_mappings = {'left-weight':0,'left-distance':1,'right-weight':2,'right-distance':3}
        category_position = 0
        
    dataset = getExamples(filename)
    (training_set, test_set) = splitByPercentage(dataset, 0.1)
    mapped_features_to_values = mapFeaturesToValues(dataset, feature_names, category_position)
    training_set_dict = splitByTargetCategory(training_set, outputs, category_position)
    
    if method == 'info_gain':        
        tree = DTree(feature_index_mappings, mapped_features_to_values)
        tree.root = tree.id3(training_set_dict, feature_names, 1)
        tree.test_set = test_set
        correct = 0
        wrong = 0
        for record in test_set:
            ans = tree.makeOneDecision(record)
            if ans == record[category_position]:
                correct += 1
            else:
                wrong += 1
        return correct,wrong
        
    
    

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 run-analysis.py 1 0 0")
        print("Where the #'s enable (1/0) analysis of:")
        print('\tgeneralization')
        print('\ttree size regarding pruning')
        print('\tROC')
        sys.exit(1)
        
    print('Running',sys.argv[0])
    analyze_generalization = int(sys.argv[1])
    analyze_tree_size = int(sys.argv[2])
    analyze_ROC_precisionrecall = int(sys.argv[3])
    
    if analyze_generalization:
        print('\nDATASET: car.data')
#        print('---- Quinlan:')
#        generalization('car.data', 'quinlan')
        print('---- Random:')
        rerr_ave_list = []
        rpave_ave_list = []
        rmave_ave_list = []
        print('calculating...')
#        for i in range(30):
#            print(i),
        e, p, m = generalization('car.data', 'random')
        rerr_ave_list.append(e)
        rpave_ave_list.append(p)
        rmave_ave_list.append(m)
        eave = 1/float(30)*sum(rerr_ave_list)
        pave = 1/float(30)*sum(rpave_ave_list) # plus ave
        mave = 1/float(30)*sum(rmave_ave_list) # minus ave
        print('Averages (err,p,m):',eave, pave, mave)
            
        print('\nDATASET: balance-scale.data')
        print('---- Quinlan:')
        generalization('balance-scale.data', 'quinlan')
        print('---- Random:')
        rerr_ave_list = []
        rpave_ave_list = []
        rmave_ave_list = []
        print('calculating...')
        for i in range(30):
            e, p, m = generalization('balance-scale.data', 'random')
            rerr_ave_list.append(e)
            rpave_ave_list.append(p)
            rmave_ave_list.append(m)
        eave = 1/float(30)*sum(rerr_ave_list)
        pave = 1/float(30)*sum(rpave_ave_list) # plus ave
        mave = 1/float(30)*sum(rmave_ave_list) # minus ave
        print('Averages (err,p,m):',eave, pave, mave)
    
    if analyze_tree_size:
        print('\nTree size analysis')
        errors = analyzeTreeSize('car.data')
        for error in errors:
            print(error)
        
    if analyze_ROC_precisionrecall:
        (c, w) = analyzeROC('car.data', 'info_gain')
        print('correct',c)
        print('wrong',w)
    
    print('\nAnalysis Complete')
    
    
    

    
    
    
    
    
    #