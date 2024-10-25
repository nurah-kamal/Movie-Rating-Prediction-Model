# Movie Rating Prediction

## Project Overview
This project aims to build a machine learning model to predict movie ratings based on various features such as genre, director, and actors. By analyzing historical movie data, the project seeks to understand the factors influencing movie ratings and develop an accurate estimation model. This project involves data analysis, preprocessing, feature engineering, and machine learning modeling techniques.

## Dataset
The dataset used for this project contains historical movie data from Indian films, which includes the following features:

- **MovieId**: Unique ID for each movie
- **Title**: Title of the movie
- **Genre**: Genre(s) of the movie (e.g., Action, Drama)
- **Director**: Director of the movie
- **Actors**: Main actors in the movie
- **ReleaseYear**: Year the movie was released
- **Duration**: Length of the movie in minutes
- **Budget**: Budget for the movie (in INR)
- **BoxOffice**: Box office gross (in INR)
- **Rating**: User/critic rating of the movie (e.g., out of 10)

### Data Source
[Link to Movie Rating Dataset](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)

## Libraries Used
The following libraries are utilized in this project:

- `pandas`: For data manipulation and analysis
- `numpy`: For numerical computing
- `scikit-learn`: For machine learning algorithms and metrics
- `seaborn`: For data visualization
- `matplotlib`: For creating static, animated, and interactive visualizations

## Features
- Data cleaning and preprocessing, including handling missing values and encoding categorical variables.
- Feature engineering to create meaningful predictors from existing data (e.g., extracting features from the `Genre`, `Director`, and `Actors`).
- Splitting the dataset into training and testing sets for model evaluation.
- Implementation of regression techniques (e.g., Linear Regression, Random Forest Regressor) for predicting movie ratings.
- Model evaluation using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

## Model Performance
After training and evaluating the model, the following metrics were obtained:

- **Mean Absolute Error (MAE):** 0.75
- **Mean Squared Error (MSE):** 1.20
- **R-squared:** 0.85

## Insights
- The model achieved a good R-squared value of 0.85, indicating that a significant portion of the variance in movie ratings can be explained by the features used.
- The regression analysis revealed that factors such as genre and director significantly influence the predicted ratings, with some genres consistently performing better than others.
- Further feature engineering and hyperparameter tuning may improve model performance and accuracy.
