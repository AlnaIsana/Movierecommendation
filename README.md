# Movie Recommendation System

## Overview

This project uses Machine Learning to recommend movies based on their similarities. The system is built using a Content-Based Filtering approach, where movie genres are analyzed and compared to suggest relevant movies to users. Natural Language Processing (NLP) techniques are used to convert genre information into numerical feature vectors, and Cosine Similarity is applied to identify movies that are most similar to the user's selected movie.

## Features

* Recommends movies based on genre similarity
* Uses Content-Based Filtering techniques
* Supports partial and case-insensitive movie search
* Uses TF-IDF Vectorization for feature extraction
* Calculates similarity using Cosine Similarity
* Provides fast and efficient movie recommendations
* Interactive user interface built with Gradio
* Can be deployed as a web application

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Gradio
* MovieLens Dataset

## Dataset

The project uses the MovieLens dataset, which contains thousands of movies along with their genre information.

Dataset Features:

* Movie ID
* Movie Title
* Genres

Example Genres:

* Action
* Adventure
* Comedy
* Drama
* Fantasy
* Romance
* Thriller
* Sci-Fi

## Project Workflow

1. Load the MovieLens dataset
2. Clean and preprocess movie data
3. Extract genre information
4. Convert genres into numerical vectors using TF-IDF Vectorizer
5. Calculate similarity scores using Cosine Similarity
6. Build the recommendation engine
7. Generate movie recommendations based on user input
8. Display recommendations through a Gradio web interface

## Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
```

## Similarity Calculation

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

## Recommendation Logic

```python
def recommend_movies(movie_name):
    matches = movies[
        movies['title'].str.contains(
            movie_name,
            case=False,
            na=False
        )
    ]

    idx = matches.index[0]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    return similarity_scores
```

## Example Recommendation

Input:

```text
Toy Story
```

Output:

```text
Toy Story 2 (1999)
Monsters, Inc. (2001)
Finding Nemo (2003)
A Bug's Life (1998)
Cars (2006)
```

## Results

The recommendation system successfully suggests movies that share similar genres and characteristics with the selected movie. By using TF-IDF Vectorization and Cosine Similarity, the model generates relevant recommendations quickly and efficiently, providing an improved movie discovery experience for users.

## Future Improvements

* Implement Collaborative Filtering
* Build a Hybrid Recommendation System
* Incorporate User Ratings and Reviews
* Use Deep Learning-based Recommendation Models
* Add Movie Posters and Metadata
* Improve Personalization using User Preferences
* Deploy on Hugging Face Spaces and Cloud Platforms

## Author

**Alna Isana**

