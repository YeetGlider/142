import pandas as pd 
import numpy as np
df = pd.read_csv('final.csv')

C = df['vote_average'].mean()
m = df['vote_count'].quantile()
q_movies = df.copy.loc[df['vote_count'] > = me]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)
q_movies[['original_title', 'vote_count', 'vote_average', 'score']].head(10)

import plotly.express as px


fig = px.bar((q_movies.head(10).sort_values('score', ascending=True)), x="score", y="original_title", orientation='h')
fig.show()

df2[['title', 'cast', 'crew', 'keywords', 'genres']].head(3)

from ast import literal_eval


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
   df2[feature] = df2[feature].apply(literal_eval)


df2.dtypes

import numpy as np

def get_director(x):
   for i in x:
       if i['job'] == 'Director':
           return i['name']
   return np.nan


df2['director'] = df2['crew'].apply(get_director)

def get_list(x):
   if isinstance(x, list):
       names = [i['name'] for i in x]
       return names
   return []


features = ['cast', 'keywords', 'genres']
for feature in features:
   df2[feature] = df2[feature].apply(get_list)

   df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)

   def clean_data(x):
   if isinstance(x, list):
       return [str.lower(i.replace(" ", "")) for i in x]
   else:
       if isinstance(x, str):
           return str.lower(x.replace(" ", ""))
       else:
           return ''


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
   df2[feature] = df2[feature].apply(clean_data)

   def create_soup(x):
   return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

def get_recommendations(title, cosine_sim):
   idx = indices[title]
   sim_scores = list(enumerate(cosine_sim[idx]))
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   sim_scores = sim_scores[1:11]
   movie_indices = [i[0] for i in sim_scores]
   return df2['title'].iloc[movie_indices]

   get_recommendations('Fight Club', cosine_sim2)

   get_recommendations('The Shawshank Redemption', cosine_sim2)

get_recommendations('The Godfather', cosine_sim2)

from google.colab import files
df2.to_csv('movies.csv')
files.download('movies.csv')
