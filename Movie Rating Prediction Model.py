import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Dataset Retrieval
file_path = 'IMDb_Movies_India.csv'  
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# IUnprocessed Data
print(data.info())
print(data.head())

#Formatting

data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')
data['Duration'] = pd.to_numeric(data['Duration'].str.replace(' min', ''), errors='coerce')

# Dropping unncessary rows and columns
data = data.dropna(subset=['Rating'])
data = data.drop(columns=['Year'])

# Separate features and target
X = data[['Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = data['Rating']

#features
numeric_features = ['Duration', 'Votes']
categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']

# Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')) ]) 

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Pipeline 
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))  # Using Random Forest Regressor
])

# Splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#checking for missing values
print("There are missing values in testing features:\n", X_test.isnull().sum())
print("There are missing values in training features: \n", X_train.isnull().sum())


# Training
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


#checking the data types 
nan_indices = np.isnan(y_pred)
if nan_indices.any():
    print("Warning: NaN values encountered in prediction.")

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")



