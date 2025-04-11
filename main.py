import numpy as np
import pandas as pd
import random

movies = [
    "The Matrix", "Inception", "Forrest Gump", "12 Angry Men", "The Lion King",
    "Cinderella", "Requiem for a Dream", "The Dark Knight", "Pulp Fiction", "Fight Club",
    "The Godfather", "The Shawshank Redemption", "The Social Network", "Interstellar", "Parasite",
    "Whiplash", "Good Will Hunting", "Joker", "The Wolf of Wall Street", "The Grand Budapest Hotel",
    "Get Out", "The Truman Show", "Black Panther", "Avengers: Endgame", "The Prestige",
    "Her", "Gladiator", "La La Land", "The Revenant", "Prisoners",
    "The Big Short", "Birdman", "Blade Runner 2049", "Shutter Island", "The Imitation Game",
    "Django Unchained", "Logan", "Arrival", "Moonlight", "Spotlight"
]

df = pd.DataFrame(columns=movies)

def random_data(n=1):
    for i in range(n):
        row = [random.randint(1, 5) if random.random() < 0.8 else np.nan for _ in range(40)]
        df.loc[len(df)] = row

def score_stats(row):
    mean_score = row.mean()
    max_score = row.max()
    min_score = row.min()

    stats = {'mean': mean_score, 'max': max_score, 'min': min_score}
    return stats

def cosine_similarity(user_1, user_2):
    # First, need to find the intersection of their ratings (Only consider movies both users have rated)
    user_1_scores = [user_1[i] for i in range(len(user_1)) if user_1[i] != -1 and user_2[i] != -1]
    user_2_scores = [user_2[i] for i in range(len(user_2)) if user_1[i] != -1 and user_2[i] != -1]

    dot_prod = np.dot(user_1_scores, user_2_scores)
    user_1_norm = np.linalg.norm(user_1_scores)
    user_2_norm = np.linalg.norm(user_2_scores)

    return dot_prod / (user_1_norm * user_2_norm)

def user_user_prediction(user, movie):
    comparing_users = df.loc[df[movie] != -1]
    cos_similarity_scores = []

    for i in range(len(comparing_users)):
        comparing_user = comparing_users.iloc[i].to_numpy()
        cos_similarity_score = cosine_similarity(user, comparing_user)

        cos_similarity_scores.append(cos_similarity_score)

    numerator = 0
    denominator = 0
    for i in range(len(comparing_users)):
        cos_sim_score = cos_similarity_scores[i]
        movie_rating = comparing_users.iloc[i][movie]

        numerator += (cos_sim_score * movie_rating)
        denominator += np.abs(cos_sim_score)

    return numerator / denominator

def item_item_prediction():
    pass

random_data(n=150)

df = df.fillna(value=-1)


