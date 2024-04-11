import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt # for visualizing the data
from wordcloud import WordCloud, STOPWORDS
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
from nltk.util import bigrams
from nltk.util import ngrams

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# import nltk
# nltk.download('punkt')  # Download the punkt tokenizer
# nltk.download('averaged_perceptron_tagger')  # Download the part-of-speech tagger (required for WordNetLemmatizer)
# nltk.download('wordnet')  # Download WordNet (required for WordNetLemmatizer)
# nltk.download('snowball_data')  # Download data for the Snowball stemmer (Porter2 stemmer)

train_df=pd.read_csv("Corona_train.csv")
test_df=pd.read_csv("Corona_validation.csv")

def calc_prior(df):
    phi={}
    total_len={}
    # res=df['Sentiment']
    n=len(df)
    for i in range (0,n):
        e=df['Sentiment'][i]
        # print('e  ',type(e),e)
        if (e in phi.keys()):
            phi[e]+=1
            total_len[e]+=(len(df['CoronaTweet'][i].split()))
        else:
            phi[e]=1
            total_len[e]=(len(df['CoronaTweet'][i].split()))
    for e in phi.keys():
        phi[e]/=n
    return phi,total_len

def create_vocab(df):
    # tweets=df['CoronaTweet'][0]
    # print(tweets)
    n=len(df)
    words=set()
    vocab={}
    vocab['Positive']={}
    vocab['Negative']={}
    vocab['Neutral']={}
    for i in range (0,n):
        tweet=df['CoronaTweet'][i]
        sent=df['Sentiment'][i]
        tokens=tweet.split()
        for token in tokens:
            words.add(token)
            if token in vocab[sent].keys():
                vocab[sent][token]+=1
            else:
                vocab[sent][token]=1
    return vocab,len(words)

def train_model(df):
    prior,total_len=calc_prior(df)
    vocab,length_of_vocab=create_vocab(df)
    theta={}
    theta['Positive']={}
    theta['Negative']={}
    theta['Neutral']={}
    for sentiment in vocab.keys():
        for words in vocab[sentiment].keys():
            theta[sentiment][words]=(vocab[sentiment][words]+1)/(total_len[sentiment]+length_of_vocab)
    # print(len(theta))
    # print(theta)
    return [theta,prior,length_of_vocab,total_len]

def test_model(test_df,train_df):
    model=train_model(train_df)
    num_of_examples=len(test_df)
    count=0
    for i in range (0,num_of_examples):
        if (test_example(test_df.loc[i],model)):
            count+=1
    return count/num_of_examples*100

def test_example(ex,model):
    tweet=ex['CoronaTweet']
    ans=ex['Sentiment']
    p1=calc_prob('Positive',tweet,model)
    p2=calc_prob('Negative',tweet,model)
    p3=calc_prob('Neutral',tweet,model)
    p=max(p1,p2,p3)
    if (p==p1 and ans=='Positive'):
        return True
    elif (p==p2 and ans=='Negative'):
        return True
    elif (p==p3 and ans=='Neutral'):
        return True
    else:
        return False

def calc_prob(sentiment,tweet,model):
    tokens=tweet.split()
    ans=np.log(model[1][sentiment])
    c=0
    for token in tokens:
        if token not in model[0][sentiment].keys():
            ans+=np.log(1/(model[2]+model[3][sentiment]))
            c+=1    
        else:
            ans+=np.log(model[0][sentiment][token])
    # print(c,len(tokens))
    return ans

# print(test_model(test_df,train_df))
#66.80838141512298
# print(test_model(train_df,train_df))
#85.04648214663004

###########CREATING WORDCLOUD ##########

def generate_wc(sentiment,model):
    wc=WordCloud(background_color='white')
    cloud_image=wc.generate_from_frequencies(model[0][sentiment])
    plt.imshow(cloud_image,interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud_'+sentiment+'_part_d.png')
    plt.show()

# model=train_model(train_df)

# generate_wc('Neutral',model)

##### RANDOM PREDICTION

def convert(sentiment):
    if (sentiment=='Positive'):
        return 1
    elif (sentiment=='Neutral'):
        return 0
    else:
        return -1
    
def random_pred_test(test_df):
    num_of_examples=len(test_df)
    count=0
    for i in range (0,num_of_examples):
        pred=random.randint(-1,1)
        ans=convert(test_df.loc[i]['Sentiment'])
        if (ans==pred):
            count+=1
    return count/num_of_examples*100

# print(random_pred_test(test_df))
## 32.91831156999696

def all_pred_positive(test_df):
    num_of_examples=len(test_df)
    count=0
    for i in range (0,num_of_examples):
        ans=convert(test_df.loc[i]['Sentiment'])
        if (ans==1):
            count+=1
    return count/num_of_examples*100

# print(all_pred_positive(test_df))
## 43.85059216519891

def confusion_matrix_example(mat,ex,model):
    tweet=ex['CoronaTweet']
    ans=ex['Sentiment']
    p1=calc_prob('Positive',tweet,model)
    p2=calc_prob('Negative',tweet,model)
    p3=calc_prob('Neutral',tweet,model)
    p=max(p1,p2,p3)
    if (p==p1):
        mat[2][convert(ans)+1]+=1
    elif (p==p2):
        mat[0][convert(ans)+1]+=1
    else:
        mat[1][convert(ans)+1]+=1
    return mat

def confusion_matrix(Train_df,Test_df):
    model=train_model(Train_df)
    mat=[[0,0,0],[0,0,0],[0,0,0]]
    m=len(Test_df)
    for i in range (0,m):
        confusion_matrix_example(mat,Test_df.loc[i],model)
    return mat

# print(confusion_matrix(train_df,test_df))
# [[905, 172, 246],
#  [55, 174, 77], 
#  [272, 271, 1121]]
# print(confusion_matrix(train_df,train_df))
# print('general')
def confusion_matrix_all_positive(Test_df):
    mat=[[0,0,0],[0,0,0],[0,0,0]]
    m=len(Test_df)
    for i in range (0,m):
        mat[2][convert(Test_df.loc[i]['Sentiment'])+1]+=1
    return mat
# print(confusion_matrix_all_positive(test_df))
# [[0, 0, 0], 
#  [0, 0, 0], 
#  [1232, 617, 1444]]
# print(confusion_matrix_all_positive(train_df))
# print('all_pos')
def confusion_matrix_random(Test_df):
    mat=[[0,0,0],[0,0,0],[0,0,0]]
    m=len(Test_df)
    for i in range (0,m):
        pred=random.randint(0,2)
        mat[pred][convert(Test_df.loc[i]['Sentiment'])+1]+=1
    return mat
# print(confusion_matrix_random(test_df))
# [[424, 211, 494],
#  [412, 221, 455],
#  [396, 185, 495]]

# print(confusion_matrix_random(train_df))
# print('random')

#### PART D
def preprocess(text):
    stemmer = PorterStemmer()
    tokens=word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    # print(' '.join(filtered_words))
    lemmatized_words = [stemmer.stem(word) for word in filtered_words]
    final_text=' '.join(lemmatized_words)
    # final_text1 = re.sub(r"[^a-zA-Z0-9]", "", final_text)
    return final_text

def process_df(name,df):
    n=len(df)
    df1=df.copy()
    for i in range (0,n):
        txt=df.loc[i]['CoronaTweet']
        df1.at[i,'CoronaTweet']=preprocess(txt)
    # file=name+'.csv'
    # df1.to_csv(file,index=False)
    return df1

# print(test_model(process_df('test',test_df),process_df('train',train_df)))
##  69.17704221075007

# print(test_model(process_df('train',train_df),process_df('train',train_df)))
##  81.74783435453202

# model_new=train_model(pd.read_csv('train.csv'))

# generate_wc('Neutral',model_new)

def preprocess_bigram(text):
    text=preprocess(text)
    words = word_tokenize(text)
# Create bigrams
    text_bigrams = list(bigrams(words))
    for bigram in text_bigrams:
        words.append(''.join(bigram))
    return ' '.join(words)

def preprocess_trigram(text):
    text=preprocess(text)
    words = nltk.word_tokenize(text)
    trigrams = list(ngrams(words, 3))
    Bigrams=list(bigrams(words))
    for bigram in Bigrams:
        words.append(''.join(bigram))
    for trigram in trigrams:
        words.append(''.join(trigram))
    return ' '.join(words)

def process_df_trigram(df):
    n=len(df)
    df1=df.copy()
    for i in range (0,n):
        txt=df.loc[i]['CoronaTweet']
        df1.at[i,'CoronaTweet']=preprocess_trigram(txt)
    return df1

# print(test_model(process_df_trigram(test_df),process_df_trigram(train_df)))
# 66.80838141512298

# print(test_model(process_df_trigram(train_df),process_df_trigram(train_df)))
# 97.93735474329178

def process_df_bigram(df):
    n=len(df)
    df1=df.copy()
    for i in range (0,n):
        txt=df.loc[i]['CoronaTweet']
        df1.at[i,'CoronaTweet']=preprocess_bigram(txt)
    return df1

# print(test_model(process_df_bigram(test_df),process_df_bigram(train_df)))
## 67.90160947464318
##v 67.38536289098087
# print(test_model(process_df_bigram(train_df),process_df_bigram(train_df)))
## 93.58493555884218
##v 84.74012254384111


### DOMAIN ADAPTATION

def train_and_test_model(source_domain,target_domain,Test_df):
    target_domain.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
    Test_df.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
    source_domain1=process_df('t',source_domain)
    target_domain1=process_df('t1',target_domain)
    Test_df1=process_df('tt',Test_df)
    Train_df=pd.concat([source_domain1,target_domain1])
    Train_df=Train_df.reset_index(drop=True)
    return test_model(Test_df1,Train_df)

# df_test=pd.read_csv('Twitter_validation.csv')
# df_1=pd.read_csv('Twitter_train_1.csv')
# df_2=pd.read_csv('Twitter_train_2.csv')
# df_3=pd.read_csv('Twitter_train_5.csv')
# df_4=pd.read_csv('Twitter_train_10.csv')
# df_5=pd.read_csv('Twitter_train_25.csv')
# df_6=pd.read_csv('Twitter_train_50.csv')
# df_7=pd.read_csv('Twitter_train_100.csv')

# print(train_and_test_model(train_df,df_1,df_test))
#46.42147117296223

# print(train_and_test_model(train_df,df_2,df_test))
# 46.98475811795891

# print(train_and_test_model(train_df,df_3,df_test))
# 48.7740225314778

# print(train_and_test_model(train_df,df_4,df_test))
# 51.027170311464545

# print(train_and_test_model(train_df,df_5,df_test))
# 51.95493704440026

# print(train_and_test_model(train_df,df_6,df_test))
# 53.61166335321405

# print(train_and_test_model(train_df,df_7,df_test))
# 54.44002650762094

# df_test.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
# df_1.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
# print(test_model(df_test,df_1))
# 37.50828363154407

# df_2.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
# print(test_model(df_test,df_2))
# 39.72829688535454

# df_3.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
# print(test_model(df_test,df_3))
# 44.36713055003313

# df_4.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
# print(test_model(df_test,df_4))
# 48.57521537442015

# df_5.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
# print(test_model(df_test,df_5))
# 49.53611663353214

# df_6.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
# print(test_model(df_test,df_6))
# 51.55732273028496

# df_7.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
# print(test_model(process_df('ll',df_test),process_df('pp',df_7)))
# 57.057654075546715

def part_f_graph():
    x=[1,2,5,10,25,50,100]
    y1=[46.42,46.98,48.77,51.02,51.95,53.61,54.44]
    y2=[37.51,39.73,44.37,48.58,49.54,51.56,57.06]
    plt.plot(x,y1,label='Domain Adaptation used')
    plt.plot(x,y2,label='Domain Adaptation not used')
    plt.xlabel('% of target data used for training')
    plt.ylabel('Validation set accuracy')
    plt.legend()
    # plt.savefig('part_1f_plot_of_accuracy.png')
    plt.show()

# part_f_graph()





