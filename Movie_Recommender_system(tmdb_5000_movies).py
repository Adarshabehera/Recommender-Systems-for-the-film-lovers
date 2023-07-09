#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head(3)


# In[4]:


credits.head(3)


# In[5]:


movies = movies.merge(credits, on = 'title')


# In[6]:


movies.head(2)


# In[7]:


movies.shape


# In[8]:


movies.columns


# In[9]:


movies = movies[['movie_id', 'genres', 'keywords','overview','title','cast','crew']]


# In[10]:


movies.isna().sum()


# In[11]:


movies.dropna(inplace =True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','Adventure','FFantasy','SciFi']


# In[15]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[16]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[17]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L  


# In[18]:


movies['genres'] = movies['genres'].apply(convert)


# In[19]:


movies['genres']


# In[20]:


movies.head(2)


# In[21]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[22]:


movies['keywords']


# In[23]:


movies.head(2)


# In[24]:


movies['cast'][0]


# In[25]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[26]:


movies['cast'] = movies['cast'].apply(convert3)


# In[27]:


movies.head(2)


# In[28]:


movies['crew'][0]


# In[29]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L  


# In[30]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[31]:


movies.head(3)


# In[32]:


movies['overview'][0]


# In[33]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[34]:


movies.head(2)


# In[35]:


## AS we are escalating the space between the words bcz of not creating confusion for model building:-

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[36]:


movies.head(2)


# In[37]:


## Giving the new_columns in the dataset for better granularity :-
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] +movies['cast'] + movies['crew']


# In[38]:


movies.head()


# In[39]:


movies['tags'][0]


# In[40]:


new_df = movies[['movie_id','title','tags']]


# In[41]:


new_df                                         ### Finally final oupur occured:-


# In[42]:


import nltk


# In[43]:


from nltk.stem.porter import PorterStemmer
ps =  PorterStemmer()


# In[44]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return "".join(y)


# In[45]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[46]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))               ## Giving the spaces bet the words:-


# In[47]:


new_df.head(2)


# In[48]:


new_df['tags'][0]


# In[49]:


new_df['tags'][1]


# In[50]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())      ## Coverting the alphabets to lower(small letterrs)recommended:-


# In[51]:


new_df.head(2)


# In[52]:


new_df.head(4)


# In[53]:


from sklearn.feature_extraction.text import CountVectorizer


# In[54]:


## Text vectorization occurs:-  (converting the tags to vectors for better granularity of data)
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[55]:


vectors = cv.fit_transform(new_df['tags']).toarray()                     ## 48060movies as compared to 5000 movies boi


# In[56]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[57]:


vectors                                      ### These all are the movies:-


# In[58]:


vectors[0]


# In[59]:


cv.get_feature_names()


# In[60]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[61]:


from sklearn.metrics.pairwise import cosine_similarity


# In[62]:


similarity = cosine_similarity(vectors)


# In[63]:


sorted(similarity[2],reverse = True)      ### This value gives 1st film distance with other 4806 films :-
                                           ## Diagonally related objects are 1
    ## Always keep one thing in your mind that sorting leads to changes the similar deviation of that ,for that we to hold all 
    ## similarity data in one place to get undisturbed.


# In[64]:


sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x:x[1])[1:6]                         ### creating the relation between distance with the indexing :-


# In[65]:


def recommend(movie):
    movies_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movies_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    


# In[66]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[67]:


recommend('Spider-Man 3')


# In[68]:


new_df.iloc[1363].title


# In[69]:


movies.head(6)


# In[70]:


### As we did the recommended part quiet well ,then our next task is to build a website :-


# In[79]:


import pickle


# In[88]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[86]:


new_df['title'].values


# In[87]:


new_df.to_dict()


# In[ ]:




