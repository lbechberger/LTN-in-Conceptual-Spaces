# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:21:34 2017

@author: nicop
"""
from MovielensTools import MovieManager


#read data
m = MovieManager()
folder = 'data_big'
movieFilePath = folder + '/movies.csv'
ratingFilePath = folder + '/ratings.csv'
tagsFilePath = folder + '/tags.csv'
m.readData(movieFilePath,ratingFilePath, tagsFilePath)

#find movies that have at least r reviews
r = 50
no_movies = 0
for movie in m.get_movies():
    if(movie.get_number_of_ratings() >= r):
        no_movies = no_movies + 1
        
print("\nNumber of movies with at least %s reviews: %s"%(r,no_movies))


#find movies that have at least t tags
t = 10
no_movies = 0
for movie in m.get_movies():
    if(movie.get_number_of_tags() >= t):
        no_movies = no_movies + 1
        
print("\nNumber of movies with at least %s tags: %s"%(t,no_movies))

for i in range(0,20):
    m1 = list(m.get_movies())[i]
    for j in range(i,20):
        m2 = list(m.get_movies())[j]
        print("Rating-Distance(%s,%s): %f"%(m1.getTitle(), m2.getTitle(), m1.compute_rating_based_distance(m2)))



for i in range(0,20):
    m1 = list(m.get_movies())[i]
    for j in range(i,20):
        m2 = list(m.get_movies())[j]
        print("Tag-Distance(%s,%s): %f"%(m1.getTitle(), m2.getTitle(), m1.compute_tag_based_distance(m2)))
