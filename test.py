import numpy as np
import pandas as pd

ratings_df = pd.read_csv("data/ratings.dat",
                         sep="::",
                         engine="python",
                         names=["UserID", "MovieID", "Rating", "Timestamp"])

users_df = pd.read_csv("data/users.dat",
                       sep="::",
                       engine="python",
                       names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])

movies_df = pd.read_csv(
                        "data/movies.dat",
                        sep="::",
                        engine="python",
                        names=["MovieID", "Title", "Genres"],
                        encoding="ISO-8859-1")

print(ratings_df.head(), '\n')
print(users_df.head(), '\n')
print(movies_df.head(), '\n')

user_1_ratings = ratings_df[ratings_df['UserID'] == 1]

first_10_movies = movies_df.loc[0:9]

movie_rating_counts = ratings_df['MovieID'].value_counts()
user_rating_counts = ratings_df['UserID'].value_counts()

def average_rating(movie_id):
    ratings = ratings_df[ratings_df["MovieID"] == movie_id]["Rating"]
    mean_rating = ratings.mean()

    return mean_rating

def num_ratings(movie_id):
    num = movie_rating_counts.get(movie_id, 0)
    return num

def user_num_ratings(user_id):
    num = user_rating_counts.get(user_id, 0)
    return num

movies_df["Mean_Rating"] = [average_rating(movie_id) for movie_id in movies_df['MovieID'].to_numpy()]
movies_df["Num_Ratings"] = [num_ratings(movie_id) for movie_id in movies_df['MovieID'].to_numpy()]
users_df["Num_Ratings"] = [user_num_ratings(user_id) for user_id in users_df['UserID'].to_numpy()]

print(users_df["Num_Ratings"].max())

