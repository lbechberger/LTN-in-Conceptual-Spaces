# -*- coding: utf-8 -*-
"""
Tools for managing films in movielens dataset

Created on Tue Nov 14 10:20:06 2017

@author: nicop
"""

import numpy as np

from matplotlib import pyplot as plt

from sklearn import manifold



"""
Read tags
"""
class TagManager:
    
    def __init__(self):
        
        self.tags = {}
      
    def read_tags(self, tagsFilePath):
        
        #read tags and put them into corresponding movies
        with open(tagsFilePath, newline='', encoding="utf8") as tagfile:
            
            #skip header
            tagfile.readline()
            
            for line in tagfile:
        
                #each line consists of user_id, movie_id, tag and timestamp
                #separated by comma 
                columns = line.split(',')
                
                tag = columns[2]
                
                if not tag in self.tags:
                    self.tags[tag] = 0
                
                #increment count by 1
                old_count = self.tags[tag]
                self.tags[tag] = old_count + 1
    
    """
    Print all tags that appear at least n times
    """            
    def printFrequent(self, n):
        
        for tag in self.tags.keys():
            count = self.tags[tag]
            if count >= n:                
                print("%s: %s"%(tag,count) )
        

"""
Read and write movies.
"""
class MovieManager:
    
    
    def __init__(self):
        
        self.movies = {}
        
        
    
    def readData(self, movieFilePath, ratingFilePath, tagsFilePath):
        
        self.read_movies(movieFilePath)
        self.read_ratings(ratingFilePath)     
        self.read_tags(tagsFilePath)
                
         
            
    def read_movies(self, movieFilePath): 
        #read movies and put them into dictionary
        with open(movieFilePath, newline='', encoding="utf8") as moviefile:
            
            #skip header
            moviefile.readline()
            
            for line in moviefile:
                
                #remove whitespaces and line breaks
                clean_line = line.rstrip()
                
                # each line consists of id, title and genres separated by commas
                columns = clean_line.split(',')
                
                #first column is id
                movie_id = columns[0]
                
                movie_title = columns[1]
                #unfortunately, some titles contain commas and need special treatment
                #we identify these cases by the number of columns
                #(usually 3 columns for id, title, genres)
                if len(columns)>3:
                    
                    for i in range(2,len(columns)-1):    
                        movie_title = movie_title+', '+columns[i]
              
                #last column contains genres separated by |
                movie_genres = columns[len(columns)-1].split('|')          
                        
                self.movies[movie_id] = Movie(movie_title, movie_genres)
        
    
    def read_ratings(self, ratingFilePath):
        
        #read ratings and put them into corresponding movies
        with open(ratingFilePath, newline='', encoding="utf8") as ratingfile:
            
            #skip header
            ratingfile.readline()
            
            for line in ratingfile:
        
                #each line consists of user_id, movie_id, rating and timestamp
                #separated by comma 
                columns = line.split(',')
                
                user_id = columns[0]
                movie_id = columns[1]
                rating = float(columns[2])
                
                self.movies[movie_id].add_rating(user_id,rating)
                
    
    def read_tags(self, tagsFilePath):
        
        #read tags and put them into corresponding movies
        with open(tagsFilePath, newline='', encoding="utf8") as tagfile:
            
            #skip header
            tagfile.readline()
            
            for line in tagfile:
        
                #each line consists of user_id, movie_id, tag and timestamp
                #separated by comma 
                columns = line.split(',')
                
                movie_id = columns[1]
                tag = columns[2]
                
                self.movies[movie_id].add_tag(tag)
                
            
    def get_movies(self):
        return self.movies.values()
                
    
    
    
class Movie:

    
     def __init__(self, title, genres):
         
         self.title = title
         self.genres = genres
         #rating maps users to their rating of this movie
         self.ratings = {}
         #tags is just a list of tags
         self.tags = list()
    
    
     def add_rating(self, user_id, rating):
        
        self.ratings[user_id] = rating
        
    
     def add_tag(self, tag):
        
        self.tags.append(tag)
        
     def getGenres(self):
        return self.genres
    
     def get_number_of_ratings(self):
         return len(self.ratings)
     
     def get_number_of_tags(self):
         return len(self.tags)
     
     
     """
     Compute distance to other movie based on rating. 
     Ratings go from 0.5 stars to 5.0 stars in 0.5-steps.
     The assumption is that movies that were ranked similar by the same users are similar.
     If the movies were not ranked both by any user, they are maximal dissimilar.
     Similarity is computed by taking all users that ranked both movies, 
     computing the squared distance and dividing by the numer of users.
     """
     def compute_rating_based_distance(self, other):
         
         users = set(self.ratings.keys()).intersection(other.ratings.keys())
         
         if len(users) == 0:
             return 25
         sum = 0
         for user in users:
             sum = sum + (self.ratings[user]-other.ratings[user])**2
             
         sum = sum / len(users)
         
         return sum
    
     """
     Compute distance to other movie based on tags. 
     The assumption is that movies that were tagged with the same tags are similar.
     We use the Jaccard distance on tags.
     """
     def compute_tag_based_distance(self, other):
         
         intersect = set(self.ratings.keys()).intersection(other.ratings.keys())
         union = set(self.ratings.keys()).union(other.ratings.keys())
         
         jac_sim = len(intersect)/len(union)
         
         return 1 - jac_sim
     
    
     def getTitle(self):
         return self.title
     
     def __str__(self):
        
        string = self.title + "\n"
        for genre in self.genres:
             string = string + " " + genre
        string = string + '\n'
        for tag in self.tags:
             string = string + " " + tag

        return string
    
    
    
    
'''
Embedd movies into real vector space using multidimensional scaling
'''
class MovieEmbedder:
    
    
     def __init__(self):
         self.dimension = 2
         self.metric = True
         self.runs = 4
         self.max_iter=3000
         self.eps=1e-6
         self.random_state=None
         self.dissimilarity="precomputed"
         self.threads=1
         
     def set_dimensions(self, dimensions):
         self.dimension = dimensions
         
     def set_no_threads(self, threads):
         self.threads = threads
    
     def set_accuracy(self, epsilon):
         self.eps = epsilon
        
     def set_no_runs(self, runs):
         self.runs = runs
        
     def set_max_iterations(self, max_iter):
         self.max_iter = max_iter
        
     def set_metric_MDS(self, metric):
         self.metric = metric
        
        
     '''
     Compute matrix consisting of vector representations of movies.
     i-th row is vector for i-th movie
     '''
     def computeRatingBasedVectorRepresentation(self, movies):
        
        distances = list()
    
        for m1 in movies:
            for m2 in movies:
                distances.append(m1.compute_rating_based_distance(m2))
        
        dist_matrix = np.array(distances)
        dist_matrix = dist_matrix.reshape(len(movies),len(movies))
                
        mds = manifold.MDS(n_components=self.dimension, 
                           metric=self.metric,
                           n_init=self.runs,
                           max_iter=self.max_iter, 
                           eps=self.eps,  
                           n_jobs=self.threads,
                           random_state=self.random_state,
                           dissimilarity=self.dissimilarity)
        
        m = mds.fit(dist_matrix).embedding_
        
        return m
      
        
     '''
     Compute matrix consisting of vector representations of movies.
     i-th row is vector for i-th movie
     '''
     def computeTagBasedVectorRepresentation(self, movies):
        
        distances = list()
    
        for m1 in movies:
            for m2 in movies:
                distances.append(m1.compute_tag_based_distance(m2))
        
        dist_matrix = np.array(distances)
        dist_matrix = dist_matrix.reshape(len(movies),len(movies))
                
        mds = manifold.MDS(n_components=self.dimension, 
                           metric=self.metric,
                           n_init=self.runs,
                           max_iter=self.max_iter, 
                           eps=self.eps,  
                           n_jobs=self.threads,
                           random_state=self.random_state,
                           dissimilarity=self.dissimilarity)
        
        m = mds.fit(dist_matrix).embedding_
        
        return m
    
     '''
     Write movie vectors in matrix line by line
     we print each movie vector once for each genre that it contains
     '''
     def printToCSVFile(self, movies, movie_matrix, filename):
         
         
         
         with open(filename, 'w+') as file:
             
             file.write('movie, genre, vector\n')
             
             for i in range(0, len(movies)):
                 
                 #prepare vector string
                 vectorString = ''
                 for j in range(0, movie_matrix.shape[1]):
                         vectorString = vectorString + str(movie_matrix[i][j])
                         if(j < movie_matrix.shape[1] - 1):
                             vectorString = vectorString + ','
                         else:
                             vectorString = vectorString + '\n'
                 
                 #print vector string for each genre
                 for genre in movies[i].getGenres():
                     
                     file.write(movies[i].getTitle() + ',' + genre + ',' + vectorString)
                        
    
    
     def plotScatterPlot(self, movies, movie_matrix, filename):
        
        plt.rcParams.update({'font.size': 7})
        plt.figure(figsize=(4000,2000))
        fig, ax = plt.subplots()
        ax.scatter(movie_matrix[:,0], movie_matrix[:,1])

        for i, movie in enumerate(movies):
            ax.annotate(movie.getTitle(), (movie_matrix[i,0],movie_matrix[i,1]))
    
        plt.savefig(filename)
        