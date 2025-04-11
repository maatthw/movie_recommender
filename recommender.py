import numpy as np
import pandas as pd
from utils import cosine_similarity

def predict_user_user_rating(user_vector, item, user_item_matrix):
    similar_users = user_item_matrix[user_item_matrix[item] != -1]
    similarities = []

    for i in range(len(similar_users)):
        other_user = similar_users.iloc[i].to_numpy()
        sim = cosine_similarity(user_vector, other_user)
        similarities.append(sim)

    numerator = 0
    denominator = 0
    for i in range(len(similar_users)):
        rating = similar_users.iloc[i][item]
        sim = similarities[i]
        numerator += sim * rating
        denominator += abs(sim)

    return numerator / denominator if denominator != 0 else np.nan

def predict_item_item_rating(user_vector, target_movie, user_item_matrix):
    user_ratings = user_vector[user_vector != -1]

    numerator = 0
    denominator = 0

    for movie_title, rating in user_ratings.items():
        if movie_title == target_movie:
            continue

        current_movie_vector = user_item_matrix[movie_title]
        target_movie_vector = user_item_matrix[target_movie]

        sim = cosine_similarity(current_movie_vector, target_movie_vector)

        numerator += sim * rating
        denominator += abs(sim)

    return numerator / denominator if denominator != 0 else np.nan

def create_user_item_matrix(ratings_df, movies_df):
    merged = ratings_df.merge(movies_df, on="MovieID")
    pivot = merged.pivot_table(index="UserID", columns="Title", values="Rating")
    return pivot.fillna(-1)