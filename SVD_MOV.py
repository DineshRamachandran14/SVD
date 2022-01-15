
# In[14]:


import numpy as np
import pandas as pd

#Import the required python libraries:


# Read the dataset from where it is downloaded in the system. It consists of two �les ‘ratings.dat’ and ‘movies.dat’ which need to be read.

# In[15]:


data = pd.io.parsers.read_csv('ratings.dat', names=['user_id', 'movie_id', 'rating', 'time'], engine='python', delimiter='::')

movie_data = pd.io.parsers.read_csv('movies.dat', names=['movie_id', 'title', 'genre'], engine='python', delimiter='::')


# In[16]:


ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)), dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values


# In[17]:


normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T


# In[18]:


A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1) 
U, S, V = np.linalg.svd(A)


# In[19]:


#Define a function to calculate the cosine similarity. Sort by most similar and return the top N results.


# In[21]:


def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude) 
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]


# In[22]:


#Define a function to print top N similar movies.


# In[23]:


def print_similar_movies(movie_data, movie_id, top_indexes): 
    print('Recommendations for {0}: \n'.format( movie_data[movie_data.movie_id == movie_id].title.values[0])) 
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])


# In[25]:


#k-principal components to represent movies, movie_id to find recommendations, top_n p 
k = 50
movie_id = 10 # (getting an id from movies.dat)
top_n = 10
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)


# In[27]:


#Print the top N similar movies.

print_similar_movies(movie_data, movie_id, indexes)





