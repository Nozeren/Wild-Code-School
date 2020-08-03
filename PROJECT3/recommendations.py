import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


#dataframes
movie4=pd.read_csv("movie4.csv")
movie_features=pd.read_csv("movie_features.csv")


#Normalizing: Normalization makes the data better conditioned for convergence.
min_max_scaler = MinMaxScaler()
movie_features2 = min_max_scaler.fit_transform(movie_features)


#Fitting Nearest Neighbors to our Data
nbrs = NearestNeighbors(n_neighbors=12, algorithm='ball_tree').fit(movie_features2)
distances, indices = nbrs.kneighbors(movie_features2)

#Helper function: get_index_from_title returns the index of the title if given the full movie name.
def get_index_from_title(title):
    return movie4[movie4["title"]==title].index.tolist()[0]


#print_similar_movies prints the top 5 similar movies after querying.
def print_similar_movies(query):
  found_id = get_index_from_title(query)
  a = movie4.loc[indices[found_id][1:]].sort_values('n_rating',ascending=False)[['title',"genres","rating"]].values.tolist()[0:5]
  return a


#your_movie returns a list of the title, genre and rating if given the full movie name.
def your_movie(item):
    b=movie4[movie4['title']==item][['title','genres','rating']].values.tolist()[0]
    return b


#pick 5 random movies by genres
def randomize(genre):
  c = movie4[(movie4['genres'].str.contains(genre,case=False)) & (movie4['rating']>3)].sample(5)[['title','genres','rating']].values.tolist()
  return c

#more https://colab.research.google.com/drive/1DYsaUk_7LXsidIt827kH6uB0BzVOPnra?usp=sharing
