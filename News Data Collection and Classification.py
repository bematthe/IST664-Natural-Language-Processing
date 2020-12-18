#Data Acquisition
import pandas as pd
import numpy as np
import twitter

api = twitter.Api(consumer_key='TbL8hrdP5Ke5jlyjUz2gEjyZs',
  consumer_secret='ODtzOxo3uei1BSLMrLj8wMu0xDXjihEK6IhafNCS624Kk1SQtP',
  access_token_key='1301219364000149504-JW61Uz4kddha4Id9RrJbKbxR6FUa1G',
  access_token_secret='NQZYXa5X1Gx6aBlssiLBwOi40voCW3I40E3RrvuHuAjgP')


topics = ['coronavirus', 'Coronavirus', 'US Elections']

twID =[]
twtext = []
twtopic = []
for tpic in topics:
    search = api.GetSearch(tpic, count = 600) # Replace happy with your search
    for tweet in search:
        twID.append(tweet.id)
        twtext.append(tweet.text)
        twtopic.append(tpic)


#Using Tweepy
import tweepy
consumer_key='TbL8hrdP5Ke5jlyjUz2gEjyZs'
consumer_secret='ODtzOxo3uei1BSLMrLj8wMu0xDXjihEK6IhafNCS624Kk1SQtP'
access_token_key='1301219364000149504-JW61Uz4kddha4Id9RrJbKbxR6FUa1G'
access_token_secret='NQZYXa5X1Gx6aBlssiLBwOi40voCW3I40E3RrvuHuAjgP'

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token_key, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth) 
language = "en"

tweetCount = 5000


topics = ['Coronavirus', 'covid', 'covid-19', 'coronavirus', 'COVID-19', 'COVID']
names = ['washingtonpost', 'nytimes', 'nbcnews']

for name in names:
    results = api.user_timeline(id=name, count=tweetCount)
    for tweet in results:
        # printing the text stored inside the tweet object
        twtext.append(tweet.text)
        twtopic.append(name)

for tpic in topics:
    search = api.search(q=tpic, lang=language) # Replace happy with your search
    for tweet in search:
        twtext.append(tweet.text)
        twtopic.append(tpic)

fd= {'Topic':twtopic, 'Text':twtext}

df = pd.DataFrame(fd)
df.to_csv('rev_new5.csv', index =0)


#tweet = pd.read_csv('rev2.csv')



#--------------------------------------------------------------------------#
#-----------------------News Articles--------------------------------------#
#--------------------------------------------------------------------------#

import newspaper
from newspaper import Article

url = ('https://www.cnn.com/us', 'https://www.msn.com/en-us/news', 'https://news.yahoo.com/us/', 'https://www.bbc.com/news', 'https://foxnews.com', 'https://msnbc.com', 'https://www.theguardian.com/us')

headlines = []
titles = []
authors = []

for u in url:
    print(u)
    cnn_paper = newspaper.build(u) 

    list_of_articles = []
    
    for artiicle in cnn_paper.articles:
        list_of_articles.append(artiicle.url)
    
    len(list_of_articles)
    
    if len(list_of_articles) > 400:
        le = 400
    else:
        le = len(list_of_articles)
    
    
    
    for i in range(0,le):
        content = Article(list_of_articles[i]) 
        content.download()
        try:
            content.parse()  
            titles.append(str(content.title))
            authors.append(str(content.authors)) 
            headlines.append(str(content.text))
        except:
            pass
    

ContentDF = pd.DataFrame({'Title': titles, 'Author': authors, 'Headlines': headlines})
ContentDF.to_csv('News_new5.csv', index = 0)


TESTDF = ContentDF

df1 = pd.read_csv('News1.csv')


df1.reset_index(drop = True, inplace = True)
df1.dropna(inplace = True)
df2 = pd.read_csv('News_new.csv')


df2.reset_index(drop = True, inplace = True)
df2.dropna(inplace = True)

ContentDF1= ContentDF

ContentDF = pd.concat([ContentDF, df1, df2])
#ContentDF.to_csv('NewsFinal.csv', index = 0)
#-------------------------------Run the code from here-------------------------#
#Make sure this file is placed in the path and make sure you update the path below:

path = 'C:/Users/dlvpr/Desktop/Data Collection/'
ContentDF = pd.read_csv(path+'NewsFinal.csv')
ContentDF.reset_index(inplace = True, drop = True)



#---------------------------------------------------------------#
#------------------News Articles--------------------------------#

newsarticles = ContentDF[['Headlines', 'Title']]
newsarticles.reset_index(inplace= True, drop = True)
newsarticles.columns



#Data pre-processing

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    
def word_cloud(text, i, stop_words):
    wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(text))
    print(wordcloud)
    fig = plt.figure(1)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Wordcloud of key')
    plt.savefig('Tweet_Group_'+str(i)+'.png')
    plt.close()
    plt.show()



import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer

ContentDF.dropna(inplace = True)
#Using NLTK & tfidftransformer

#Cleaning the data set and prepare the corpus
#Prepare the corpus
corpus = []
for i in range(0, len(ContentDF['Title'])):    
    text = ContentDF['Headlines'][i]
    ##Convert to list from string
    text = text.split()
    text = " ".join(text)
    corpus.append(text)


#Tokenize

textwords = []

for i in corpus:
    textwords.extend([word.lower() for sent in nltk.sent_tokenize(i) for word in nltk.word_tokenize(sent)])

tok = pd.DataFrame({'words': textwords})
print ('there are ' + str(tok.shape[0]) + ' items in Corpus')





# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)


#unigrams
textdist = FreqDist(textwords)
textitems = textdist.most_common(50)
for item in textitems:
    print (item[0], '\t', item[1])


#Lets look at the bigrams
textdist = FreqDist((nltk.bigrams(textwords)))
textitems = textdist.most_common(50)
for item in textitems:
    print (item[0], '\t', item[1])

#Lets look at the trigrams
textdist = FreqDist((nltk.trigrams(textwords)))
textitems = textdist.most_common(50)
for item in textitems:
    print (item[0], '\t', item[1])




#find total word count
wordcount = {}

for i in ContentDF['Headlines']:
    i = i.lower()
    for w in nltk.word_tokenize(i):
        if w not in wordcount.keys():
            wordcount[w] = 1
        else:
            wordcount[w] +=1
            
#find word count by document.
doccount ={}
for ww in wordcount.keys():
    doccount[ww] = 0
for i in ContentDF['Headlines']:
    i = i.lower()
    f = nltk.word_tokenize(i)
    for wordss in wordcount.keys():
        if wordss in f:
            doccount[wordss] += 1
            
            
#remove words that appear in less than 3 documents.

iter1dict = {}

for key, value in doccount.items():
    if value > 5:
        iter1dict[key] = value

#remove terms that are smaller than 3 characters
iter2dict = {}

for key, value in iter1dict.items():
    if len(key) >3:
        iter2dict[key] = value



#Clean the tweets
        
cleantweet = ""
coll = iter2dict.keys()
f = {}
j = {}
o = {}



for index, row in newsarticles.iterrows():
    rtoken = nltk.word_tokenize(row[0])
    for w in rtoken:
        if w in iter2dict.keys():
            cleantweet = cleantweet + " " + w
    f[index] = cleantweet
    j[index] = row[1]
    o[index] = row[0]
    cleantweet = ""


texts = list(f.values())
sentis = list(j.values())
origtext = list(o.values())

dff = {'Original Text': origtext, 'Clean Text':texts, 'Title':sentis}




#Cleaned text
newdf1= pd.DataFrame(dff)


newdf1.head()


import re
#Remove Stopwords and special characters

def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False


nltkstopwords = nltk.corpus.stopwords.words('english')
morestopwords = ['know', 'https', 'http', 'well', 'said', 'one', 'time', 'people', 'look', 'many', 'ago', 'even', 'much', 'didnt', 'see', 'weve', 'say', 'ive', 'got', 'come', 'like', 'thats', 'ever', 'theyre', 'going', 'dont', 'want', 'rrthe', '\r', 'shall', 'made', 'et.', 'al', 'could','would','might','must','need',
                 'rrrrrr','rr', 'h','b', 'sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve", 
                 "n't", 'readingrrcoronavirus', 'treatmentrbut', 'recoveryrbookr22.95rview', "image", "reuters", "caption", "breaking", "news", "via", "via image caption", "copy", "copyright", "getty", "nbc", "cnn", "images", "using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]

stopwords = nltkstopwords + morestopwords

#More special characters to be removed
junkchars = ['x99', 'x9d','\\xe2\\x80\\x94','\\n', '\\','xe2','x80','x94',
             'x9c', 'x93', 'r20', 'r26', '.r', '\r\n']

#Remove the stopwords
te = {}
f = {}
j = {}
o = {}
from nltk.tokenize.treebank import TreebankWordDetokenizer

for index, row in newdf1.iterrows():
    textwords = nltk.word_tokenize(row[1])
    textcl = [w for w in textwords if w not in stopwords if not alpha_filter(w)]
    #Remove the html characters 
    textcleantokens = []
    for d in textcl:
        for u in junkchars: 
            d=d.replace(u,'')
#        d=lemmatizer.lemmatize(d)
        #d=englishStemmer.stem(d)
        textcleantokens.append(d)
    textcleantokens = [w for w in textcleantokens if w not in stopwords if not alpha_filter(w)]
    te[index] = TreebankWordDetokenizer().detokenize(textcleantokens)
    f[index] = row[0]
    j[index] = row[2]



texts = list(f.values())
sentis = list(j.values())
fintext = list(te.values())

dff = {'Original Text': texts, 'Clean Text':fintext, 'Title':sentis}
#Cleaned text
newdf= pd.DataFrame(dff)

newdf.head()

newdf.dropna(inplace = True)
#----------------------------------------------------------------------------#
#Verify the cleansed dataset

corpus = []
for i in range(0, len(newdf['Title'])):    
    text = newdf['Clean Text'][i]
    ##Convert to list from string
    text = text.split()
    text = " ".join(text)
    corpus.append(text)


textwords = []

for i in corpus:
    textwords.extend([word.lower() for sent in nltk.sent_tokenize(i) for word in nltk.word_tokenize(sent)])

tok = pd.DataFrame({'words': textwords})
print ('there are ' + str(tok.shape[0]) + ' items in Corpus')

#unigrams
textdist = FreqDist(textwords)
textitems = textdist.most_common(50)
for item in textitems:
    print (item[0], '\t', item[1])


#Lets look at the bigrams
textdist = FreqDist((nltk.bigrams(textwords)))
textitems = textdist.most_common(50)
for item in textitems:
    print (item[0], '\t', item[1])

#Lets look at the trigrams
textdist = FreqDist((nltk.trigrams(textwords)))
textitems = textdist.most_common(50)
for item in textitems:
    print (item[0], '\t', item[1])




#-------------------------------------------------------------------------#
#Feature extraction
#First Iteration: Lets do Unigram
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
 
newdf = newdf.dropna()

X = newdf['Clean Text']

#Using tfidf vectorizer
from sklearn.metrics.pairwise import cosine_similarity
X = newdf['Clean Text']

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=2500,
                                min_df=0.1, stop_words='english',
                                 use_idf=True, ngram_range=(1,2))


tfidf_matrix = tfidf_vectorizer.fit_transform(X) 
#fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()

from sklearn.cluster import KMeans
num_clusters = 7
km = KMeans(n_clusters=num_clusters)
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

#Save your model to pickle file
from sklearn.externals import joblib

joblib.dump(km,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

#Topic Modelling and word clouds

artic = { 'News Article': texts,  'Title': sentis, 'cluster': clusters, 'clean content': fintext }
frame = pd.DataFrame(artic, index = [clusters] , columns = ['News Article', 'Title', 'cluster', 'clean content'])
frame['cluster'].unique()


#Topic Modelling for Clusters

#Use LDA - Latent Dirichelet Allocation

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
from gensim import corpora, models

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        result.append(token)
    return result

j=[]
topicdict = []
topiclust = []
for i in np.sort(frame['cluster'].unique()):
    if frame['cluster'][i].count() > 20:
        processed_docs = frame['clean content'][i].map(preprocess)
        dictionary = gensim.corpora.Dictionary(processed_docs)    
        dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)    
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]    
        tfidf = models.TfidfModel(bow_corpus)
    
    
        #TFIDF for corpus
        corpus_tfidf = tfidf[bow_corpus]
    
        temtp = []
    #Using TFIDF Corpus
        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=1, id2word=dictionary, passes=2, workers=4)
        for idx, topic in lda_model_tfidf.print_topics(-1):
            for r in lda_model_tfidf.show_topic(idx, topn=10):
                temtp.append(r[0])
        j.append(i)
        topicdict.append(temtp)
        topiclust.append('Topic of Cluster'+str(i))
        

topicmodel = {'Cluster':topiclust, 'Topic': topicdict}

topicdf = pd.DataFrame(topicmodel)




import matplotlib.pyplot as plt


cont = frame['clean content']
clss = frame['cluster']
names = [] 
sds = []
leng = []
arra = []
ffs = []
ww=0
for i in j:
    arra = []
    names.append('Cluster'+str(i))
    leng.append(len(frame[frame['cluster']==i]['clean content']))
    dff = frame[frame['cluster']==i]['clean content']
    for yy in range(len(frame[frame['cluster']==i]['clean content'])):
        arra.append(dff[yy])
    ffs.append(arra)
    word_cloud(arra, ww, stopwords)
    ww=ww+1


#Visualize the data
import matplotlib as mpl
from sklearn.manifold import MDS
MDS()
#Calculate distances for plotting

dist = 1 - cosine_similarity(tfidf_matrix)

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
print()
print()

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#7570b3', 6: '#e7298a'}
#set up cluster names using a dict
cluster_names = {0: 'C1', 
                 1: 'C2', 
                 2: 'C3', 
                 3: 'C4', 
                 4: 'C5',
                 5: 'C6',
                 6: 'C7'}
#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=texts) )
#group by cluster
groups = df.groupby('label')
# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], size=8)      
plt.show() #show the plot


#-------------------------------------------------------------------------------#
#-------------------------------Lets Train a model------------------------------#
#-------------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
 
newdf = newdf.dropna()
#Test Train Split
X = frame['clean content']
y = frame['cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

tf_vectorizer=CountVectorizer() 
X_train_tf = tf_vectorizer.fit_transform(X_train)
X_test_tf = tf_vectorizer.transform(X_test)

tf_tfidf = TfidfTransformer()
tf_tfidf.fit(X_train_tf)
X_train_tf_idf = tf_tfidf.transform(X_train_tf)

tf_tfidf.fit(X_test_tf)
X_test_tf_idf = tf_tfidf.transform(X_test_tf)

from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train_tf_idf, y_train)
predictions = text_classifier.predict(X_test_tf_idf)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

