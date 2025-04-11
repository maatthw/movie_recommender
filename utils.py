import numpy as np

def compute_movie_stats(ratings_df, movies_df):
    movie_rating_counts = ratings_df['MovieID'].value_counts()
    movies_df["Mean_Rating"] = movies_df['MovieID'].apply(
        lambda movie_id: ratings_df[ratings_df["MovieID"] == movie_id]["Rating"].mean()
    )
    movies_df["Num_Ratings"] = movies_df['MovieID'].apply(
        lambda movie_id: movie_rating_counts.get(movie_id, 0)
    )
    return movies_df


def compute_user_stats(ratings_df, users_df):
    user_rating_counts = ratings_df['UserID'].value_counts()
    users_df["Num_Ratings"] = users_df['UserID'].apply(
        lambda user_id: user_rating_counts.get(user_id, 0)
    )
    return users_df


def cosine_similarity(vec1, vec2):
    common = [(a, b) for a, b in zip(vec1, vec2) if a != -1 and b != -1]
    if not common:
        return 0
    v1, v2 = zip(*common)
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_prod / (norm_v1 * norm_v2) if norm_v1 != 0 and norm_v2 != 0 else 0