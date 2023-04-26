import argparse #This line of code imports the "argparse" module, which provides a convenient way to parse command line arguments in Python.
import json #This line of code imports the "json" module, which provides functions for working with JSON data 
import numpy as np #This line of code imports the "numpy" module, which provides support for working with arrays and numerical operations
import pandas as pd #This line of code imports the "pandas" module, which provides data structures and functions for working with tabular data
from sklearn.metrics.pairwise import cosine_similarity #This line of code imports the "cosine_similarity" function from the "sklearn.metrics.pairwise" module, which provides support for calculating pairwise similarity between vectors using the cosine similarity measure.
from sklearn.model_selection import train_test_split
#This line of code imports the "train_test_split" function from the "sklearn.model_selection" module, which provides support for splitting data into training and testing sets for machine learning models.
from sklearn.feature_extraction.text import TfidfVectorizer
#This line of code imports the "TfidfVectorizer" class from the "sklearn.feature_extraction.text" module, which provides support for extracting text features using the Term Frequency-Inverse Document Frequency (TF-IDF) weighting scheme.
from sklearn.neighbors import KNeighborsClassifier
#This line of code imports the "KNeighborsClassifier" class from the "sklearn.neighbors" module, which provides support for building K-Nearest Neighbors (K-NN) classification models.




def read_file():
    #This line of code defines a function called "read_file" that takes no arguments.
    df = pd.read_json('data/yummly.json')
    #This line of code reads the JSON data from the "yummly.json" file using the "pd.read_json" method and assigns the resulting DataFrame to the "df" variable.
    return df

def clean_data(data):
    # convert to lower case
    data = [d.lower() for d in data]
    # remove punctuation
    data = [d.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')) for d in data]
    # remove numbers
    data = [d.translate(str.maketrans('', '', '1234567890')) for d in data]
    # remove extra whitespaces
    data = [d.strip() for d in data]
    return data

def vectorize_form(df, args):
    # Clean input ingredients
    input_ingredients = clean_data(args.ingredient)

    # Clean and join all ingredients in DataFrame
    ingredients_list = [' '.join(clean_data(ingredients)) for ingredients in df['ingredients']]

    # Append input ingredients to the list of ingredients
    data = ingredients_list + [' '.join(input_ingredients)]

    return data
# Define a function that takes a pandas DataFrame, a list of recipe ingredients, and additional parameters as input and returns the predicted cuisine label, predicted cuisine probability score, and information on the closest N recipes based on cosine similarity scores.
def predict_cuisine(df, data, args):
    # Create an instance of TfidfVectorizer with an ngram range of 1-2 and remove stop words using 'english'.
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    # Transform the list of ingredients (excluding the last item) into a sparse matrix of TF-IDF features.
    ingredient_features = tfidf_vectorizer.fit_transform(data[:-1])
    # Transform the last item in the list of ingredients into a sparse matrix of TF-IDF features.
    input_ = tfidf_vectorizer.transform([data[-1]])
    # Create an instance of KNeighborsClassifier with args.N neighbors and fit it to the ingredient_features sparse matrix and corresponding cuisine labels in the DataFrame df.
    classifier = KNeighborsClassifier(n_neighbors=args.N).fit(ingredient_features, df['cuisine'])
    # Predict the cuisine label of the recipe based on the input_ sparse matrix of features.
    pred = classifier.predict(input_)
    # Calculate the maximum predicted probability of the predicted cuisine label.
    pred_cuisine_score = np.max(classifier.predict_proba(input_))
    # Calculate the cosine similarity between the input_ sparse matrix of features and the ingredient_features sparse matrix of features and flatten the resulting array.
    scores = cosine_similarity(input_, ingredient_features).ravel()
    # Create a list of the args.N closest recipes based on cosine similarity scores. For each of the closest recipes, return the corresponding id from the DataFrame df and the cosine similarity score rounded to 3 decimal places.
    closest_n_data = [(df['id'][i], round(scores[i], 3)) for i in np.argsort(scores)[::-1][:args.N]]
    # Return the predicted cuisine label, predicted cuisine probability score, and information on the closest N recipes based on cosine similarity scores.
    return pred, round(pred_cuisine_score, 3), closest_n_data


def output(pred, pred_cuisine_score, n_closest):
    # Convert any int64 type objects to int type
    n_closest = [(int(id_), score) for id_, score in n_closest]
    # Construct the JSON object
    output = {'cuisine': pred[0],
              'score': float(pred_cuisine_score),
              'closest': [{'id': id_, 'score': score} for id_, score in n_closest]}
    print(json.dumps(output, indent=4))


def main(parser):
    args = parser.parse_args()
    df = read_file()
    cleaned_data = clean_data(df)
  #  corpus = cleaned_data(df, args)
    corpus = vectorize_form(df, args)
    pred, pred_cuisine_score, n_closest = predict_cuisine(df, corpus, args)
    output(pred, pred_cuisine_score, n_closest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict cuisine from ingredients')
    parser.add_argument("--N", type=int, help='Number of neighbors')
    parser.add_argument("--ingredient", type=str, action='append', required=True, help="List of ingredients")
    main(parser)


 
