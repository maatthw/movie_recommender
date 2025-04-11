from data_loader import load_data
from utils import compute_movie_stats, compute_user_stats
from recommender import create_user_item_matrix, predict_user_user_rating
import pandas as pd

ratings, users, movies = load_data("data/ratings.dat", "data/users.dat", "data/movies.dat")

movies = compute_movie_stats(ratings, movies)
users = compute_user_stats(ratings, users)

user_item_matrix = create_user_item_matrix(ratings, movies)

# Testing

user_1_vector = user_item_matrix.loc[1]
unrated_movies = user_1_vector[user_1_vector == -1].index

predictions = {}
for movie in unrated_movies[:5]:
    prediction = predict_user_user_rating(user_1_vector, movie, user_item_matrix)
    predictions[movie] = prediction

print("Top Predictions for User 1:")
for title, pred in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
    print(f"{title}: {pred:.2f}")
