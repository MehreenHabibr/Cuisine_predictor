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
    #This line of code defines a function called "output" that takes in three arguments: "pred", "pred_cuisine_score", and "n_closest".
    output = {#This line of code initializes a dictionary object called "output".
        "cuisine": pred[0],#This line of code adds a key-value pair to the "output" dictionary, where the key is "cuisine" and the value is the first element of the "pred" list.
        "score": round(pred_cuisine_score, 3), #This line of code adds a key-value pair to the "output" dictionary, where the key is "score" and the value is the rounded "pred_cuisine_score" value.
        "closest": [{"id": tup[0], "score": round(tup[1], 3)} for tup in n_closest]
    }#This line of code adds a key-value pair to the "output" dictionary, where the key is "closest" and the value is a list comprehension that creates a list of dictionaries. Each dictionary in the list contains an "id" key and a "score" key, where the "id" key corresponds to the first element of a tuple in the "n_closest" list, and the "score" key corresponds to the rounded second element of the same tuple.
    print(json.dumps(output, indent=4))
    #This line of code prints the "output" dictionary object in a JSON-formatted string with an indentation level of 4.



def main(parser):
    args = parser.parse_args()
    df = read_file()
    corpus = vectorize_form(df, args)
    pred, pred_cuisine_score, n_closest = predict_cuisine(df, corpus, args)
    output(pred, pred_cuisine_score, n_closest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict cuisine from ingredients')
    parser.add_argument("--N", type=int, help='Number of neighbors')
    parser.add_argument("--ingredient", type=str, action='append', required=True, help="List of ingredients")
    main(parser)

 
