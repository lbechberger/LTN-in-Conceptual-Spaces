# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:21:34 2017

@author: nicop
"""
from MovielensTools import MovieManager
from MovielensTools import MovieEmbedder


#read data
m = MovieManager()
folder = 'data_big'
movieFilePath = folder + '/movies.csv'
ratingFilePath = folder + '/ratings.csv'
tagsFilePath = folder + '/tags.csv'
m.readData(movieFilePath,ratingFilePath, tagsFilePath)

#find movies that have at least r reviews
r = 15000
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


my_genres = ['Action', 'Adventure', 'Children', 'Fantasy', 'Horror']
genreToMovies = {}
genreToIDs = {}

for genre in my_genres:
    genreToMovies[genre] = list()
    genreToIDs[genre] = list()

for i in range(0, len(filtered_movies)):
    for genre in my_genres:
        if genre in filtered_movies[i].getGenres():
            genreToMovies[genre].append(filtered_movies[i])
            genreToIDs[genre].append(i)
    
for genre in my_genres:
    print(genre)
    embedder.plotScatterPlot(genreToMovies[genre],m[genreToIDs[genre],:], genre+'.png')


