import pandas as pd


def load_data(ratings_path, users_path, movies_path):
    ratings_df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )

    users_df = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    )

    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="ISO-8859-1"
    )

    return ratings_df, users_df, movies_df