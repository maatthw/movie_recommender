import numpy as np
import pandas as pd
from utils import cosine_similarity

def predict_user_user_rating(user_vector, item, df):
    similar_users = df[df[item] != -1]
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

def create_user_item_matrix(ratings_df, movies_df):
    merged = ratings_df.merge(movies_df, on="MovieID")
    pivot = merged.pivot_table(index="UserID", columns="Title", values="Rating")
    return pivot.fillna(-1)