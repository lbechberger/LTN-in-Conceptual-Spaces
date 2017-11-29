# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:21:34 2017

@author: nicop
"""
from MovielensTools import MovieManager
from MovielensTools import MovieEmbedder


#read data
m = MovieManager()
folder = 'data_small'
movieFilePath = folder + '/movies.csv'
ratingFilePath = folder + '/ratings.csv'
tagsFilePath = folder + '/tags.csv'
m.readData(movieFilePath,ratingFilePath, tagsFilePath)

#find movies that have at least r reviews
r = 100
no_movies = 0
for movie in m.get_movies():
    if(movie.get_number_of_ratings() >= r):
        no_movies = no_movies + 1
        
print("\nNumber of movies with at least %s reviews: %s"%(r,no_movies))


#find movies that have at least t tags
t = 1000
no_movies = 0
for movie in m.get_movies():
    if(movie.get_number_of_tags() >= t):
        no_movies = no_movies + 1
        
print("\nNumber of movies with at least %s tags: %s"%(t,no_movies))



embedder = MovieEmbedder()
embedder.set_dimensions(2)

filtered_movies = list()
for movie in m.get_movies():
    if(movie.get_number_of_ratings() >= r):
        filtered_movies.append(movie)

m = embedder.computeRatingBasedVectorRepresentation(filtered_movies)
embedder.printToCSVFile(filtered_movies, m, 'vectors.csv')
#embedder.plotScatterPlot(filtered_movies,m, 'plot.png')

