# Collaborative Filtering Movie Recommender

## Overview
**Collaborative filtering** to recommend movies to users based on their ratings. The system calculates user similarity using three different methods:
- **Cosine Similarity**
- **Pearson Correlation Coefficient**
- **Spearman Rank Correlation**

Similarity is calculated manually.

A simple web application allows users to interact with the recommender system by:
- Receiving movie recommendations based on user similarity.
- Choosing which similarity metric to use for recommendations.
- Recalculating similarity scores dynamically.
- Adding new ratings to update recommendations.

## Features
### Admin can:
- **force recalculation** of the movie recommendations if users added new ratings.
- **choose the similarity metric** to be used for the recommendations.
- **choose number of most similar users** to be considered for the recommendations.
### Users can:
- **look at the recommendations** based on the ratings given by the other users.
- **add ratings** to the movies they have watched.
- **get updated recommendations** based on the ratings they have given.

## Installation
### Clone the Repository
```sh
git clone https://github.com/yourusername/collaborative-filtering-recommender.git
cd collaborative-filtering-recommender
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage
### Running the Web App
```sh
python app.py
```
This will start the web server, and you can access the app in your browser at `http://localhost:5000` (or another specified port).
