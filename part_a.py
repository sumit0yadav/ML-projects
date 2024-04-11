'''
This is started code for part a. 
Using this code is OPTIONAL and you may write code from scratch if you want
'''
import heapq
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from numpy import log2
import pickle
import os
import time
import matplotlib.pyplot as plt
neg_inf=float('-inf')
data_type={}
class DTNode:

    def __init__(self, depth,x_data,y_data,X_val,Y_val, is_leaf = False, column = None):
        self.parent=None
        self.x=x_data
        self.y=y_data
        self.depth = depth
        self.children = []
        self.is_leaf = is_leaf
        self.col=column
        self.val_of_node=None
        self.type_of_feature_split_on=None
        self.predicted_class=None
        self.x_val=X_val
        self.y_val=Y_val
        self.correct_val=-1
        self.correct_val_subtree=-1
        self.other_correct_val=0

    def __lt__(self,other):
        return self.correct_val+self.other_correct_val-self.correct_val_subtree>other.correct_val+other.other_correct_val-other.correct_val_subtree

    def count_nodes(self):
        temp=self
        if (temp.is_leaf):
            return 1
        else:
            ans=1
            for child in temp.children:
                ans+=child.count_nodes()
            return ans

    def get_children(self, X):
        '''
        Args:
            X: A single example np array [num_features]
        Returns:
            child: A DTNode
        '''
        temp=self
        while (temp.is_leaf==False):
            found=False
            med=0
            if temp.type_of_feature_split_on!='cat':
                med=np.median(temp.x[:,temp.col])
            for child in temp.children:
                if temp.type_of_feature_split_on=='cat':
                    if child.val_of_node==X[temp.col]:
                        temp=child
                        found=True
                        break
                else:
                    if X[temp.col]<med and child.val_of_node==0:
                        temp=child
                        found=True
                        break
                    elif X[temp.col]>=med and child.val_of_node==1:
                        temp=child
                        found=True
                        break
            if (found==False):
                return temp
        return temp

    def calc_mutual_info(self,feature):
        X=self.x[:,feature]
        m=X.shape[0]
        new_nodes={}
        arr_zero=self.y == 0
        zero_count=np.count_nonzero(arr_zero)
        p=zero_count/m
        if p==0 or p==1:
            entropy=0
        else:
            entropy=(-p*log2(p)-(1-p)*log2(1-p))
        if data_type[feature]=='cont':
            # print(feature)
            med=np.median(X)
            new_nodes={0:[0,0],1:[0,0]}
            for i in range (0,m):
                if X[i]<med:
                    new_nodes[0][self.y[i]]+=1
                else:
                    new_nodes[1][self.y[i]]+=1
            if new_nodes[0][0]+new_nodes[0][1]==0:
                p0=0
            else:
                p0=new_nodes[0][0]/(new_nodes[0][0]+new_nodes[0][1])
            if new_nodes[1][0]+new_nodes[1][1]==0:
                p1=0
            else:
                p1=new_nodes[1][0]/(new_nodes[1][0]+new_nodes[1][1])
            if p0==0 or p0==1:
                ent0=0
            else:
                ent0=(-p0*log2(p0)-(1-p0)*log2(1-p0))
            if p1==0 or p1==1:
                ent1=0
            else:
                ent1=(-p1*log2(p1)-(1-p1)*log2(1-p1))
            w0=(new_nodes[0][0]+new_nodes[0][1])/m
            return entropy-(w0*ent0+(1-w0)*ent1)
        else:
            for i in range (0,m):
                if X[i] not in new_nodes.keys():
                    new_nodes[X[i]]=[0,0]
                    new_nodes[X[i]][self.y[i]]+=1
                else:
                    new_nodes[X[i]][self.y[i]]+=1
            p_array=[]
            w_array=[]
            ent_arr=[]
            for node in new_nodes.keys():
                p_array.append(new_nodes[node][0]/(new_nodes[node][0]+new_nodes[node][1]))
                w_array.append((new_nodes[node][0]+new_nodes[node][1])/m)
            for p,w in zip(p_array,w_array):
                if p==1 or p==0:
                    ent_arr.append(0)
                else:
                    ent_arr.append(w*(-p*log2(p)-(1-p)*log2(1-p)))
            cond_entropy=sum(ent_arr)
            return entropy-cond_entropy

    def choose_Attr_to_split(self):
        columns=self.x.shape[1]
        max_info=neg_inf
        ans=-1
        for i in range (0,columns):
            info=self.calc_mutual_info(i)
            if max_info<info:
                ans=i
                max_info=info
        return ans

    def split_node(self,feature):
        self.col=feature
        self.is_leaf=False
        X=self.x[:,feature]
        X_val=self.x_val[:,feature]
        m=X.shape[0]
        ind_dict={}
        ind_dict_val={}
        if data_type[feature]=='cont':
            self.type_of_feature_split_on='cont'
            med=np.median(X)
            # med_val=np.median(X_val)
            ind_dict[0]=np.where(X<med)
            ind_dict[1]=np.where(X>=med)
            ind_dict_val[0]=np.where(X_val<med)
            ind_dict_val[1]=np.where(X_val>=med)
            for i in range (0,2): 
                if len(ind_dict[i][0])==0:
                    continue   
                node= DTNode(self.depth+1,self.x[ind_dict[i]],self.y[ind_dict[i]],self.x_val[ind_dict_val[i]],self.y_val[ind_dict_val[i]],is_leaf=True)
                self.children.append(node)
                node.parent=self
                node.val_of_node=i
                if 2*sum(node.y)>=node.y.shape[0]:
                    node.predicted_class=1
                else:
                    node.predicted_class=0
                node.correct_val=np.count_nonzero( node.y_val == node.predicted_class)
            if len(self.children)==1:
                if self.children[0].val_of_node==0:
                    self.children[0].other_correct_val= np.count_nonzero(self.y[ind_dict_val[1]] == self.children[0].predicted_class)
                else:
                    self.children[0].other_correct_val= np.count_nonzero(self.y[ind_dict_val[0]] == self.children[0].predicted_class)
                
        else:
            self.type_of_feature_split_on='cat'
            values=np.unique(X)
            for value in values:
                ind_dict[value]=np.where(X == value)
                ind_dict_val[value]=np.where(X_val == value)
            for value in values:
                node=DTNode(self.depth+1,self.x[ind_dict[value]],self.y[ind_dict[value]],self.x_val[ind_dict_val[value]],self.y_val[ind_dict_val[value]],is_leaf=True)
                self.children.append(node)
                node.parent=self
                node.val_of_node=value
                if 2*sum(node.y)>=node.y.shape[0]:
                    node.predicted_class=1
                else:
                    node.predicted_class=0
                node.correct_val=np.count_nonzero( node.y_val == node.predicted_class)
            if len(self.children)==1:
                if self.children[0].val_of_node==0:
                    self.children[0].other_correct_val= np.count_nonzero(self.y[ind_dict_val[1]] == self.children[0].predicted_class)
                else:
                    self.children[0].other_correct_val= np.count_nonzero(self.y[ind_dict_val[0]] == self.children[0].predicted_class)
        return
    
    def GrowTree(self,max_depth):
        if self.depth>=max_depth:
            return
        temp=self
        # print('cur depth',self.depth)
        best_attr=temp.choose_Attr_to_split()
        temp.split_node(best_attr)
        for child in temp.children:
            child.GrowTree(max_depth)
        return 
              
    def get_correct_val(self):
        temp=self
        if temp.correct_val_subtree!=-1:
            return temp.correct_val_subtree
        if temp.is_leaf:
            temp.correct_val_subtree=temp.correct_val+temp.other_correct_val
            return temp.correct_val_subtree
        else:
            ans=temp.other_correct_val
            for child in temp.children:
                ans+=child.get_correct_val()
            temp.correct_val_subtree=ans
            return ans

    def get_tree_nodes(self,l):
            temp=self
            # l.append(temp)
            if temp.is_leaf:
                return
            else:
                for child in temp.children:
                    l.append(child)
                    child.get_tree_nodes(l)
                return
        
class DTTree:
    def __init__(self):
        #Tree root should be DTNode
        self.root = None   
        self.val_acc=[]  
        self.test_acc=[]
        self.train_acc=[]  
        self.num_nodes=[]

    def fit(self, X, y, types,X_val1=np.array([]),Y_val1=np.array([]), max_depth = 45):
        '''
        Makes decision tree
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            None
        '''
        self.root= DTNode(0,X,y,X_val=X_val1,Y_val=Y_val1,is_leaf=True)
        self.root.GrowTree(max_depth)
        return

    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        predictions=[]
        m=X.shape[0]
        for i in range (0,m):
            # print('i done',i)
            leaf=self.root.get_children(X[i])
            predictions.append(leaf.predicted_class)
        return np.array(predictions)

    def post_prune(self):
        nodes=[]
        x_train,y_train=read_file_One_hot_encoding('train.csv')
        x_test,y_test=read_file_One_hot_encoding('test.csv')
        print('pruning....')
        nodes.append(self.root)
        self.root.get_tree_nodes(nodes)
        heapq.heapify(nodes)
        deleted_nodes=[]
        while len(nodes)>0 and nodes[0].correct_val+nodes[0].other_correct_val-nodes[0].correct_val_subtree>=0:
            # print('ins')
            curr=heapq.heappop(nodes)
            if curr in deleted_nodes:
                continue
            curr.get_tree_nodes(deleted_nodes)
            curr.children=[]
            diff=curr.correct_val_subtree
            curr.correct_val_subtree=curr.correct_val+curr.other_correct_val
            curr.type_of_feature_split_on=None
            curr.col=None
            curr.is_leaf=True
            temp=curr.parent
            while temp!=None:
                # deleted_nodes.append(temp)
                temp.correct_val_subtree+=(curr.correct_val_subtree-diff)
                temp=temp.parent
            heapq.heapify(nodes)
            # self.train_acc.append(self.calc_accuracy(x_train,y_train.flatten()))
            # self.test_acc.append(self.calc_accuracy(x_test,y_test.flatten()))
            # self.val_acc.append(self.root.correct_val_subtree)
            # self.num_nodes.append(self.root.count_nodes())
        print('prune acc',self.calc_accuracy(self.root.x_val,self.root.y_val))
        return

    def calc_accuracy(self,X,y):
        # print('testing...')
        pred=self(X)
        # print('pred done...')
        return np.sum( y == pred)/X.shape[0]*100
    
def read_file(file_path):
    df=pd.read_csv(file_path)
    arr=df.to_numpy()
    train_y=arr[:,13]
    arr1=np.delete(arr,0,axis=1)
    train_x= np.delete(arr1, 12, axis=1)
    return train_x,train_y

label_encoder = None 
def read_file_One_hot_encoding(filepath):
    global label_encoder
    data = pd.read_csv(filepath)
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(),y.to_numpy()
d=45

# x_train,y_train=read_file('train.csv')
# x_test,y_test=read_file('test.csv')
# print('all pos train',100*np.count_nonzero(y_train)/y_train.shape[0],100-100*np.count_nonzero(y_train)/y_train.shape[0])
# print('all pos test',100*np.count_nonzero(y_test)/y_test.shape[0],100-100*np.count_nonzero(y_test)/y_test.shape[0])

def plot_1a(x,y_test,y_train=None,y_val=None):
    # x=[5,10,15,20,25]
    # y1=[57.807,59.876,59.876,59.876,59.876]
    # y2=[88.769,99.681,99.77,99.77,99.77]
    plt.plot(x,y_test,label='Testing accuracy',marker='o')
    if (y_train!=None):
        plt.plot(x,y_train,label='Training accuracy',marker='o')
    if (y_val!=None):
        plt.plot(x,y_val,label='Validation accuracy',marker='o')
    plt.xlabel('maximum depth allowed in Decision tree')
    # Value of ccp_alpha
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('part_1b_plot_of_accuracy.png')
    plt.show()
    return

def part_1c():
    test_x_ohe,test_y_ohe=read_file_One_hot_encoding('test.csv')
    train_x_ohe,train_y_ohe=read_file_One_hot_encoding('train.csv')
    val_x_ohe,val_y_ohe=read_file_One_hot_encoding('val.csv')
    if os.path.isfile(os.path.join('.', 'dec_tree_prune_'+str(d)+'.pickle')):
        print('model already trained for pruning for d = ',d)
        print('d',d)
        model = pickle.load(open('dec_tree_prune_'+str(d)+'.pickle', 'rb'))
    else:
        global data_type
        data_type = ['cont']*train_x_ohe.shape[1]
        model=DTTree()
        print('training start...')
        st=time.time()
        model.fit(train_x_ohe,train_y_ohe.flatten(),data_type,val_x_ohe,val_y_ohe.flatten())
        model.root.get_correct_val()
        model.post_prune()
        et=time.time()
        print('train time for prune',et-st)
        print('d',d)
        pickle.dump((model), open('dec_tree_prune_'+str(d)+'.pickle', 'wb'))
    print('num nodes for prune',str(d),model.root.count_nodes())
    print('val  for prune',model.calc_accuracy(val_x_ohe,val_y_ohe.flatten()))
    # val=[i/val_y_ohe.shape[0]*100 for i in model.val_acc]
    # print(val[0],'valll')
    print('test',model.calc_accuracy(test_x_ohe,test_y_ohe.flatten()))
    print('train',model.calc_accuracy(train_x_ohe,train_y_ohe.flatten()))
    # plot_1a(model.num_nodes,model.test_acc,model.train_acc,val)

# part_1c()

###                   final initial 
##num nodes for d = 15 303  16321
##num nodes for d = 25 303  51721
##num nodes for d = 35 303  87121
##num nodes for d = 45 185  122521

### val accuracy
# d = 15, 70.344
# d = 25, 70.344
# d = 35, 70.344
# d = 45, 70.344

### test accuracy
# d = 15, 60.599
# d = 25, 60.599
# d = 35, 60.599
# d = 45, 60.599

### train accuracy
# d = 15, 63.204
# d = 25, 63.204
# d = 35, 63.204
# d = 45, 63.204


## data_type = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]
def part1a():
    test_x,test_y=read_file('test.csv')
    train_x,train_y=read_file('train.csv')
    print(train_x.shape,train_y.shape,'kkk')
    if os.path.isfile(os.path.join('.', 'dec_tree_'+str(d)+'.pickle')):
        print('hehe')
        print('d',d)
        model = pickle.load(open('dec_tree_'+str(d)+'.pickle', 'rb'))
    else:
        data_type = ['cat','cat','cat',"cont","cat","cat","cat","cat","cat" ,"cont","cont" ,"cont" ]
        model=DTTree()
        print('training start...')
        st=time.time()
        model.fit(train_x,train_y,data_type)
        et=time.time()
        print('train time',st-et)

        print('d',d)
        pickle.dump((model), open('dec_tree_'+str(d)+'.pickle', 'wb'))

    print(model.calc_accuracy(train_x,train_y))


### test accuracy
# d = 5, 57.8076525336091
# d = 10, 59.87590486039297
# d = 15, 59.87590486039297
# d = 20, 59.87590486039297
# d = 25, 59.87590486039297

### train accuracy
# d = 5, 88.76964354158682
# d = 10, 99.68059281972658
# d = 15, 99.77002683020314
# d = 20, 99.77002683020314
# d = 25, 99.77002683020314


# plot_1a()

### ONE HOT ENCODING
def part1b():
    print('ohe...')
    train_x_ohe,train_y_ohe=read_file_One_hot_encoding('train.csv')
    test_x_ohe,test_y_ohe=read_file_One_hot_encoding('test.csv')
    val_x_ohe,val_y_ohe=read_file_One_hot_encoding('val.csv')

    if os.path.isfile(os.path.join('.','dec_tree_'+str(d)+'_ohe1.pickle')):
        print('ohe model already trained for d = ',d)
        model = pickle.load(open('dec_tree_'+str(d)+'_ohe1.pickle', 'rb'))
    else:
        global data_type
        data_type = ['cont']*train_x_ohe.shape[1]
        model=DTTree()
        print('training start for ohe...')
        st=time.time()
        model.fit(train_x_ohe,train_y_ohe.flatten(),data_type,val_x_ohe,val_y_ohe)
        et=time.time()
        print('train time for ohe model for d = ',d,et-st)
        pickle.dump((model), open('dec_tree_'+str(d)+'_ohe1.pickle', 'wb'))
    print('num nodes in ohe model for d = ',str(d),model.root.count_nodes())
    print('ohe test accuracy',model.calc_accuracy(val_x_ohe,val_y_ohe.flatten()))
    x1=[5,15,25,35,45]
    acc_test=[59.462,58.014,58.014,58.014,58.014]
    acc_train=[61.824,98.556,98.594,98.594,98.594]
    acc_val=[57.356,57.356,57.356,57.356,57.356]
    plot_1a(x1,acc_test,acc_train,acc_val)
part1b()


### val accuracy
# d = 15, 57.35632183908046
# d = 25, 57.35632183908046
# d = 35, 57.35632183908046
# d = 45, 57.35632183908046

### test accuracy
# d = 5, 59.4622543950362
# d = 15, 58.014477766287484
# d = 25, 58.014477766287484
# d = 35, 58.014477766287484
# d = 45, 58.014477766287484

### train accuracy
# d = 5, 61.82445381372174
# d = 15, 98.55627954516417
# d = 25, 98.59460840679698
# d = 35, 98.59460840679698
# d = 45, 98.59460840679698


def part1d_a():
    train_x,train_y=read_file_One_hot_encoding('train.csv')
    test_x,test_y=read_file_One_hot_encoding('test.csv')
    val_x,val_y=read_file_One_hot_encoding('val.csv')
    # List of max_depth values to test
    max_depth_values = [15, 25, 35, 45]
    # Iterate through different max_depth values
    for max_depth in max_depth_values:
        print('training start for',str(max_depth),'...')
        dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        dt_classifier.fit(train_x, train_y)
        predictions_train = dt_classifier.predict(train_x)
        predictions_test = dt_classifier.predict(test_x)
        predictions_val = dt_classifier.predict(val_x)
        accuracy_train = accuracy_score(train_y, predictions_train)*100
        accuracy_test = accuracy_score(test_y,predictions_test)*100
        accuracy_val = accuracy_score(val_y,predictions_val)*100
        print('Train Accuracy for d = ',str(max_depth),'is ',accuracy_train)
        print('Test Accuracy for d = ',str(max_depth),'is ',accuracy_test)
        print('Val Accuracy for d = ',str(max_depth),'is ',accuracy_val)

# part1d_a()

# Train Accuracy for d =  15 is  71.40666922192412
# Test Accuracy for d =  15 is  60.39296794208894
# Val Accuracy for d =  15 is  58.85057471264368
# Train Accuracy for d =  25 is  85.49891401558708
# Test Accuracy for d =  25 is  63.28852119958634
# Val Accuracy for d =  25 is  60.45977011494252
# Train Accuracy for d =  35 is  94.45509135045356
# Test Accuracy for d =  35 is  65.4601861427094
# Val Accuracy for d =  35 is  62.758620689655174
# Train Accuracy for d =  45 is  99.54005366040629
# Test Accuracy for d =  45 is  63.49534643226473
# Val Accuracy for d =  45 is  61.03448275862069

# acc_train=[71.406,85.498,94.455,99.54]
# acc_test=[60.393,63.288,65.46,63.495]
# acc_val=[58.85,60.459,62.758,61.034]
# x=[15,25,35,45]
# plot_1a(x,acc_test,acc_train,acc_val)

## final model depth = 35

def part1d_b():
    train_x,train_y=read_file_One_hot_encoding('train.csv')
    test_x,test_y=read_file_One_hot_encoding('test.csv')
    val_x,val_y=read_file_One_hot_encoding('val.csv')
    alpha_values = [0.001, 0.01, 0.1, 0.2]
    for alpha in alpha_values:
        print('training start for',str(alpha),'...')
        dt_classifier = DecisionTreeClassifier(max_depth=35,ccp_alpha=alpha)
        dt_classifier.fit(train_x, train_y)
        predictions_train = dt_classifier.predict(train_x)
        predictions_test = dt_classifier.predict(test_x)
        predictions_val = dt_classifier.predict(val_x)
        accuracy_train = accuracy_score(train_y, predictions_train)*100
        accuracy_test = accuracy_score(test_y,predictions_test)*100
        accuracy_val = accuracy_score(val_y,predictions_val)*100
        print('Train Accuracy for ccp_alpha = ',str(alpha),'is ',accuracy_train)
        print('Test Accuracy for ccp_alpha = ',str(alpha),'is ',accuracy_test)
        print('Val Accuracy for ccp_alpha = ',str(alpha),'is ',accuracy_val)

# part1d_b()

# training start for 0.001 ...
# Train Accuracy for ccp_alpha =  0.001 is  68.21259741918998
# Test Accuracy for ccp_alpha =  0.001 is  64.11582213029989
# Val Accuracy for ccp_alpha =  0.001 is  64.36781609195403
# training start for 0.01 ...
# Train Accuracy for ccp_alpha =  0.01 is  50.33857161108982
# Test Accuracy for ccp_alpha =  0.01 is  49.638055842812825
# Val Accuracy for ccp_alpha =  0.01 is  47.356321839080465
# training start for 0.1 ...
# Train Accuracy for ccp_alpha =  0.1 is  50.33857161108982
# Test Accuracy for ccp_alpha =  0.1 is  49.638055842812825
# Val Accuracy for ccp_alpha =  0.1 is  47.356321839080465
# training start for 0.2 ...
# Train Accuracy for ccp_alpha =  0.2 is  50.33857161108982
# Test Accuracy for ccp_alpha =  0.2 is  49.638055842812825
# Val Accuracy for ccp_alpha =  0.2 is  47.356321839080465

# x=[0.001, 0.01, 0.1, 0.2]
# acc_train=[68.212,50.338,50.338,50.338]
# acc_test=[64.116,49.638,49.638,49.638]
# acc_val=[64.369,47.356,47.356,47.356]
# plot_1a(x,acc_test,acc_train,acc_val)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def part1e():
    train_x,train_y=read_file_One_hot_encoding('train.csv')
    test_x,test_y=read_file_One_hot_encoding('test.csv')
    val_x,val_y=read_file_One_hot_encoding('val.csv')
    print('rf...')
    param_grid = {
        'n_estimators': [50, 150, 250, 350],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'min_samples_split': [2, 4, 6, 8, 10]
    }
    # rf_classifier = RandomForestClassifier(n_estimators=250,min_samples_split=10,max_features=1.0, random_state=42)
    # grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='accuracy')
    # grid_search.fit(train_x,train_y.flatten())
    # best_params = grid_search.best_params_
    # best_oob_accuracy = grid_search.best_estimator_.oob_score_
    # print("Best Parameters:", best_params)
    # print("Best OOB Accuracy:", best_oob_accuracy)
    # print('train',accuracy_score(train_y.flatten(),rf_classifier.predict(train_x))*100)
    # print('test',accuracy_score(test_y.flatten(),rf_classifier.predict(test_x))*100)
    # print('val',accuracy_score(val_y.flatten(),rf_classifier.predict(val_x))*100)
    val=[]
    train=[]
    test=[]
    for n in param_grid['min_samples_split']:
        rf_classifier = RandomForestClassifier(n_estimators=250,min_samples_split=n,max_features=1.0, random_state=42)
        rf_classifier.fit(train_x,train_y.flatten())
        val.append(accuracy_score(val_y.flatten(),rf_classifier.predict(val_x))*100)
        train.append(accuracy_score(train_y.flatten(),rf_classifier.predict(train_x))*100)
        test.append(accuracy_score(test_y.flatten(),rf_classifier.predict(test_x))*100)
 
    print('test',test)
    print('train',train)
    print('val',val)
    plot_1a(param_grid['min_samples_split'],test,train,val)


# part1e()



# Best Parameters: {'max_features': 1.0, 'min_samples_split': 10, 'n_estimators': 250}
# Best OOB Accuracy: 0.7186661556151782
# train 96.85703334610962
# test 72.69906928645294
# val 71.03448275862068


## n estimators
#    test =[72.18200620475697, 72.59565667011375, 72.69906928645294, 73.11271975180972]
#     train =[96.3970870065159, 96.89536220774244, 96.85703334610962, 96.84425705889869]
#     val =[70.6896551724138, 71.37931034482759, 71.03448275862068, 71.37931034482759]

## min_samples_split
# test [71.66494312306101, 72.38883143743536, 72.59565667011375, 73.2161323681489, 72.69906928645294]
# train [100.0, 99.94889485115625, 99.14398875686726, 98.21131979046889, 96.85703334610962]
# val [69.3103448275862, 69.88505747126436, 70.6896551724138, 70.6896551724138, 71.03448275862068]

## max_features
# test [70.21716649431231, 71.25129265770424, 72.28541882109617, 72.59565667011375, 72.80248190279214, 72.69906928645294]
# train [93.61185639453174, 96.05212725182062, 96.55040245304714, 96.85703334610962, 97.0231250798518, 96.85703334610962]
# val [69.08045977011494, 69.08045977011494, 70.34482758620689, 70.34482758620689, 70.91954022988506, 71.03448275862068]

# if __name__ == '__main__':

    #change the path if you want
    # X_train,y_train = get_np_array('train.csv')
    # X_test, y_test = get_np_array("test.csv")

    # max_depth = 10
    # tree = DTTree()
    # tree.fit(X_train,y_train,types, max_depth = max_depth)