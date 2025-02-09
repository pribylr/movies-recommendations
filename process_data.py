import numpy as np
import pandas as pd
import pickle
from itertools import islice


def load_and_process_data():
    movies = pd.read_csv('data/Netflix_Dataset_Movie.csv')
    ratings = pd.read_csv('data/ratings.csv')
    ratings = ratings.drop(['Unnamed: 0'], axis=1)
    return movies, ratings


def id_valid(user_id: int, data: pd.DataFrame) -> bool:
    """
    Function to check if user_id is in the database
    """

    if user_id.isdigit():
        user_id = int(user_id)
    else:
        return False
    if user_id in data['User_ID'].unique() or user_id == 0:
        return True
    else:
        return False


def find_common_ratings(id1: int, id2: int, user_movie_ratings: list):
    """
    Function to get ratings of two users and their common ratings
    :param user_movie_ratings: list of dictionaries with user ratings
    """
    ratings1 = user_movie_ratings[id1-1]
    ratings2 = user_movie_ratings[id2-1]
    common_ratings = {k: (ratings1[k], ratings2[k]) for k in ratings1 if k in ratings2}
    return ratings1, ratings2, common_ratings


def compute_pearson_sim(id1: int, id2: int, user_movie_ratings: list):
    """
    Function to compute Pearson similarity between two users
    :param id1: identifier of first user
    :param id2: identifier of second user
    :param user_movie_ratings: list of dictionaries with user ratings (movie -> rating)
    :return: Pearson similarity coefficient
    """
    ratings1, ratings2, common_ratings = find_common_ratings(id1, id2, user_movie_ratings)
    if len(common_ratings) < 70:
        return None
    mean1 = sum(ratings1.values())/len(ratings1.values())
    mean2 = sum(ratings2.values())/len(ratings2.values())

    nominator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    for rating1, rating2 in common_ratings.values():
        nominator += ( (rating1 - mean1)*(rating2 - mean2) )
        denominator1 += np.square(rating1 - mean1)
        denominator2 += np.square(rating2 - mean2)
    denominator = np.sqrt(denominator1*denominator2)
    if denominator == 0:
        return None
    result = nominator/denominator

    return result


def compute_spearman_sim(id1: int, id2: int, user_movie_ratings: list):
    """
    Function to compute Spearman similarity between two users
    :param id1: identifier of first user
    :param id2: identifier of second user
    :param user_movie_ratings: list of dictionaries with user ratings (movie -> rating)
    :return: Spearman similarity coefficient
    """
    ratings1, ratings2, common_ratings = find_common_ratings(id1, id2, user_movie_ratings)
    if len(common_ratings) < 70:
        return None
    sorted_ratings1 = sorted(set([ratings[0] for ratings in common_ratings.values()]), reverse=True)
    sorted_ratings2 = sorted(set([ratings[1] for ratings in common_ratings.values()]), reverse=True)
    rank_dict1 = {rating: rank for rank, rating in enumerate(sorted_ratings1, start=1)}
    rank_dict2 = {rating: rank for rank, rating in enumerate(sorted_ratings2, start=1)}
    rank1 = [rank_dict1[rating[0]] for rating in common_ratings.values()]
    rank2 = [rank_dict2[rating[1]] for rating in common_ratings.values()]
    distances = [np.square(r1 - r2) for r1, r2 in zip(rank1, rank2)]
    distances = sum(distances)
    n = len(common_ratings)
    result = 1 - ((6 * distances) / (n * (np.square(n) - 1)))
    return result


def compute_cosine_sim(id1: int, id2: int, user_movie_ratings: list):
    """
    Function to compute Cosine similarity between two users
    :param id1: identifier of first user
    :param id2: identifier of second user
    :param user_movie_ratings: list of dictionaries with user ratings (movie -> rating)
    :return: Cosine similarity coefficient
    """
    ratings1, ratings2, common_ratings = find_common_ratings(id1, id2, user_movie_ratings)
    if len(common_ratings) < 70:
        return None
    nominator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    for rating1, rating2 in common_ratings.values():
        nominator += (rating1*rating2)
    for rating in ratings1.values():
        denominator1 += np.square(rating)
    for rating in ratings2.values():
        denominator2 += np.square(rating)
    denominator = np.sqrt(denominator1)*np.sqrt(denominator2)
    result = nominator/denominator
    return result


def load_similarities(similarity: str):
    if similarity == 'pearson':
        with open('data/pearson_sims.pkl', 'rb') as file:
            sim_list = pickle.load(file)
    elif similarity == 'spearman':
        with open('data/spearman_sims.pkl', 'rb') as file:
            sim_list = pickle.load(file)
    else:
        with open('data/cosine_sims.pkl', 'rb') as file:
            sim_list = pickle.load(file)
    return sim_list


def save_similarities(similarity: str, sim_list: list):
    if similarity == 'pearson':
        with open('data/pearson_sims.pkl', 'wb') as file:
            pickle.dump(sim_list, file)
    elif similarity == 'spearman':
        with open('data/spearman_sims.pkl', 'wb') as file:
            pickle.dump(sim_list, file)
    else:
        with open('data/cosine_sims.pkl', 'wb') as file:
            pickle.dump(sim_list, file)


def load_user_movie_ratings():
    with open('data/user_movie_ratings.pkl', 'rb') as file:
        user_movie_ratings = pickle.load(file)
    return user_movie_ratings


def save_user_movie_ratings(user_movie_ratings: list):
    with open('data/user_movie_ratings.pkl', 'wb') as file:
        pickle.dump(user_movie_ratings, file)


def load_movies_db():
    with open('data/movies_db.pkl', 'rb') as file:
        movies_db = pickle.load(file)
    return movies_db


def load_recommendations(similarity: str):
    if similarity == 'pearson':
        with open('data/pearson_rec.pkl', 'rb') as file:
            recommendations = pickle.load(file)
    elif similarity == 'spearman':
        with open('data/spearman_rec.pkl', 'rb') as file:
            recommendations = pickle.load(file)
    else:
        with open('data/cosine_rec.pkl', 'rb') as file:
            recommendations = pickle.load(file)

    return recommendations


def save_recommendations(similarity: str, recommendations: dict, usr_id: int):
    if similarity == 'pearson':
        with open('data/pearson_rec.pkl', 'wb') as file:
            pickle.dump(recommendations, file)
    elif similarity == 'spearman':
        with open('data/spearman_rec.pkl', 'wb') as file:
            pickle.dump(recommendations, file)
    else:
        with open('data/cosine_rec.pkl', 'wb') as file:
            pickle.dump(recommendations, file)


def find_users_with_updated_ratings(similarity: str):
    """
    Function to find users with recently updated ratings
    :return list of user id's
    """
    res = []
    tmp = pd.read_csv('data/changed.csv')
    changed = False
    for index, row in tmp.iterrows():
        if similarity == 'pearson':
            if row['P'] == 1:
                res.append(row['user_id'])
                tmp.loc[tmp['user_id'] == row['user_id'], 'P'] = 0
                changed = True
        elif similarity == 'spearman':
            if row['S'] == 1:
                res.append(row['user_id'])
                tmp.loc[tmp['user_id'] == row['user_id'], 'S'] = 0
                changed = True
        else:
            if row['C'] == 1:
                res.append(row['user_id'])
                tmp.loc[tmp['user_id'] == row['user_id'], 'C'] = 0
                changed = True
    if changed:
        tmp.to_csv('data/changed.csv', index=False)
    return res


def recalculate_similarity(similarity: str, ids: list):
    """
    Function to recalculate similarity for users with updated ratings
    :param similarity: one of three similarity coefficients
    :param ids: list of user id's with updated ratings
    :return:
    """
    with open('data/user_movie_ratings.pkl', 'rb') as file:
        user_movie_ratings = pickle.load(file)
    if similarity == 'pearson':
        with open('data/pearson_sims.pkl', 'rb') as file:
            sim_list = pickle.load(file)
    elif similarity == 'spearman':
        with open('data/spearman_sims.pkl', 'rb') as file:
            sim_list = pickle.load(file)
    else:
        with open('data/cosine_sims.pkl', 'rb') as file:
            sim_list = pickle.load(file)

    for usr_id in ids:
        for i in range(len(user_movie_ratings)):
            if i != usr_id:
                if similarity == 'pearson':
                    sim = compute_pearson_sim(usr_id, i, user_movie_ratings)
                elif similarity == 'spearman':
                    sim = compute_spearman_sim(usr_id, i, user_movie_ratings)
                else:
                    sim = compute_cosine_sim(usr_id, i, user_movie_ratings)
                if sim is not None:
                    sim_list[usr_id][i] = sim
                    sim_list[i][usr_id] = sim

    if similarity == 'pearson':
        with open('data/pearson_sims.pkl', 'wb') as file:
            pickle.dump(sim_list, file)
    elif similarity == 'spearman':
        with open('data/spearman_sims.pkl', 'wb') as file:
            pickle.dump(sim_list, file)
    else:
        with open('data/cosine_sims.pkl', 'wb') as file:
            pickle.dump(sim_list, file)


def make_predictions(user_movie_ratings: list, user_ratings: dict, user_similarities: dict, user_id: int,
                     movie_id: int, how_many_users: int):
    """
    Function to calculate prediction for a single movie for a single user
    """
    prediction = 0.0
    kappa = 0.5
    rated_counter = 0  # counter of users that rated movie with id 'movie_id'

    for key, value in user_similarities.items():  # user_similarities is sorted from highest similarity first
        similar_user_ratings = user_movie_ratings[key-1]
        if movie_id in similar_user_ratings:
            current_movie_rating = similar_user_ratings[movie_id]
            close_user_mean_rating = sum(similar_user_ratings.values()) / len(similar_user_ratings)
            rated_counter += 1
            prediction += value * (current_movie_rating - close_user_mean_rating)
        if rated_counter >= how_many_users:
            break
    prediction *= kappa
    prediction += sum(user_ratings.values()) / len(user_ratings)
    return prediction


def find_top_predictions(prediction_dict: dict, movies: pd.DataFrame):
    """
    Function to find top 10 movies with the highest predicted rating
    :return list of 10 movie names
    """
    prediction_dict = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))
    highest_predictions = dict(islice(prediction_dict.items(), 10))
    recommendation_movies_id = [tmp for tmp in highest_predictions.keys()]
    names = []
    for movie_id in recommendation_movies_id:
        row = movies[movies['Movie_ID'] == movie_id]
        names.append(row['Name'].iloc[0])
    return names


def create_recommendation(how_many_users: int, user_id: int, sim_list: list, user_movie_ratings: list):
    """
    Function that goes through all movies and makes predictions
    :param how_many_users:
    :param user_id:
    :param sim_list:
    :param user_movie_ratings:
    :return:
    """
    user_similarities = sim_list[user_id-1]
    user_similarities = dict(sorted(user_similarities.items(), key=lambda item: item[1], reverse=True))
    user_ratings = user_movie_ratings[user_id-1]
    movies = load_movies_db()
    res = {}
    for movie_id in movies['Movie_ID']:
        # make prediction only for movies the user has not rated
        if movie_id not in user_ratings:
            prediction = make_predictions(user_movie_ratings, user_ratings, user_similarities, user_id, movie_id, how_many_users)
            res[movie_id] = prediction
    recommendation = find_top_predictions(res, movies)
    return recommendation


def calculate_new_recommendations(similarity: str, how_many_users: int, changed_users: list):
    """
    Function that goes through all users with updated ratings and calculates new recommendations
    :param similarity: similarity identifier string
    :param changed_users: list of user id's with updated ratings
    """
    sim_list = load_similarities(similarity)
    user_movie_ratings = load_user_movie_ratings()
    recommendations = load_recommendations(similarity)
    target = 0
    for usr_id in changed_users:
        recommendation = create_recommendation(how_many_users, usr_id, sim_list, user_movie_ratings)
        recommendations[usr_id] = recommendation
        target = usr_id
    save_recommendations(similarity, recommendations, target)


def recalculate_recommendations(similarity: str, how_many_users: int):
    """
    Function to recalculate recommendations for users with updated ratings
    :param similarity: similarity identifier string
    :param how_many_users: number of users to take into account when making predictions
    """
    changed_users = find_users_with_updated_ratings(similarity)
    if len(changed_users) != 0:
        recalculate_similarity(similarity, changed_users)
        calculate_new_recommendations(similarity, how_many_users, changed_users)


def find_names_of_movies(user_id: int):
    """
    Function to find names of movies rated by user based on ID's of the movies
    :param user_id: int
    :return: dictionary {movie_name: rating}
    """
    with open('data/user_movie_ratings.pkl', 'rb') as file:
        user_movie_ratings = pickle.load(file)
    user_ratings = user_movie_ratings[user_id - 1]
    with open('data/movies_db.pkl', 'rb') as file:
        movies_db = pickle.load(file)

    res = {movies_db.iloc[movie_id]['Name']: rating for movie_id, rating in user_ratings.items()}
    return res


def find_not_rated_movies(user_id: int):
    """
    Function to find movies that user has not rated
    :param user_id: identifier of user
    :return: list of strings - movies names
    """
    with open('data/movies_db.pkl', 'rb') as file:
        movies_db = pickle.load(file)
    with open('data/user_movie_ratings.pkl', 'rb') as file:
        user_movie_ratings = pickle.load(file)
    user_ratings = user_movie_ratings[user_id - 1]
    res = []
    for movie_id in movies_db['Movie_ID']:
        if movie_id not in user_ratings.keys():
            res.append(movies_db.iloc[movie_id]['Name'])
    return res


def add_rating(user_id: int, movie_name: str, rating: int):
    """
    Function to add rating
    :param user_id: identifier of user who added rating
    :param movie_name: name of movie that was rated
    :param rating: value of rating
    """
    with open('data/movies_db.pkl', 'rb') as file:
        movies_db = pickle.load(file)
    movie_id = (movies_db[movies_db['Name'] == movie_name]['Movie_ID']).iloc[0]

    with open('data/user_movie_ratings.pkl', 'rb') as file:
        user_movie_ratings = pickle.load(file)

    user_movie_ratings[user_id - 1][movie_id] = rating
    with open('data/user_movie_ratings.pkl', 'wb') as file:
        pickle.dump(user_movie_ratings, file)


def change_user_flag(user_id: int):
    """
    Function to change flag that signals that an user updated rating
    :param user_id: identifier of user
    """
    tmp = pd.read_csv('data/changed.csv')
    tmp.loc[tmp['user_id'] == user_id, 'P'] = 1
    tmp.loc[tmp['user_id'] == user_id, 'S'] = 1
    tmp.loc[tmp['user_id'] == user_id, 'C'] = 1
    tmp.to_csv('data/changed.csv', index=False)
