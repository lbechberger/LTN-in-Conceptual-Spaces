# -*- coding: utf-8 -*-
"""
Tools for managing films in movielens dataset

Created on Tue Nov 14 10:20:06 2017

@author: nicop
"""


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
                
                # each line consists of id, title and genres separated by commas
                columns = line.split(',')
                
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