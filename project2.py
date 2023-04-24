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
    with open('data/yummly.json', 'r') as file:
    #This line of code opens the "yummly.json" file in read mode using a context manager and assigns the resulting file object to the "file" variable.
        data = json.load(file)
    #This line of code reads the contents of the "file" object using the "json.load" method and assigns the resulting JSON data to the "data" variable.
    df = pd.DataFrame(data)
    #This line of code creates a Pandas DataFrame from the "data" dictionary object and assigns it to the "df" variable.
    return df
    #This line of code returns the "df" DataFrame from the function.


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
    #This line of code defines a function called "vectorize_form" that takes in two arguments: "df" and "args".
    input_ingredients = clean_data(args.ingredient)
    #This line of code calls the "clean_data" function on the "args.ingredient" input and assigns the resulting cleaned data to the "input_ingredients" variable.
    ingredients_list = df['ingredients']
    #This line of code retrieves the "ingredients" column from the "df" DataFrame and assigns it to the "ingredients_list" variable.
    data = list(map(' '.join, ingredients_list))
    #This line of code uses the "map" function to join each list of ingredients in "ingredients_list" with a space separator and returns a list of strings. The resulting list is assigned to the "data" variable.
    data = clean_data(data)
    #This line of code calls the "clean_data" function on the "data" list of strings and assigns the resulting cleaned data to the "data" variable.
    data.append(' '.join(input_ingredients))
    #This line of code appends a string to the end of the "data" list that consists of the cleaned input ingredients joined with a space separator.
    return data
   #This line of code returns the "data" list from the function.

def predict_cuisine(df, data, args):
    n = args.N
    labels = df['cuisine']
    ids = df['id'].tolist()
    n_closest = []
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    #This line of code initializes a TfidfVectorizer object with a specified ngram range of 1 to 2 and with English stop words.
    ingredient_features = tfidf_vectorizer.fit(data[:-1]).transform(data[:-1])
    #This line of code fits the TfidfVectorizer object to the "data" list except for the last element (i.e., data[:-1]) and then transforms it into a matrix of TfidfVectorizer features. The resulting matrix is assigned to a variable called "ingredient_features".
    input_ = tfidf_vectorizer.transform([data[-1]])
    #This line of code transforms the last element of the "data" list (i.e., data[-1]) into a TfidfVectorizer feature matrix. The resulting matrix is assigned to a variable called "input_".
    classifier = KNeighborsClassifier(n_neighbors=args.N)
   # This line of code initializes a KNeighborsClassifier object with a specified number of neighbors (i.e., "args.N") and assigns it to a variable called "classifier".
    classifier.fit(ingredient_features, labels)
    #This line of code fits the KNeighborsClassifier object to the "ingredient_features" matrix and the "labels" vector, which represent the feature matrix and the corresponding cuisine labels, respectively.
    pred = classifier.predict(input_)
    #This line of code uses the trained KNeighborsClassifier object to predict the cuisine label of the input feature matrix (i.e., "input_"). The predicted label is assigned to a variable called "pred".
    pred_cuisine_score = classifier.predict_proba(input_)[0].max()
    #This line of code calculates the maximum predicted probability score for the predicted cuisine label (i.e., "pred") and assigns it to a variable called "pred_cuisine_score".
    scores = cosine_similarity(input_, ingredient_features, dense_output=False).toarray().ravel()
    #This line of code calculates the cosine similarity scores between the input feature matrix (i.e., "input_") and the "ingredient_features" matrix. The resulting scores are flattened into a one-dimensional array and assigned to a variable called "scores".
    sort_scores = np.argsort(scores)[::-1]
   # This line of code sorts the cosine similarity scores in descending order and returns their corresponding indices. The resulting indices are assigned to a variable called "sort_scores".
    closest_n_indices = sort_scores[:n]
    #This line of code selects the top "n" indices with the highest cosine similarity scores and assigns them to a variable called "closest_n_indices".
    closest_n_data = [(ids[i], round(scores[i], 3)) for i in closest_n_indices]
    #This line of code creates a list of tuples, where each tuple contains the "id" value and the cosine similarity score for a selected index in the "closest_n_indices" list. The resulting list of tuples is assigned to a variable called "closest_n_data".

    return pred, round(pred_cuisine_score, 3), closest_n_data
#This line of code returns three values: the predicted cuisine label (i.e., "pred"), the maximum predicted probability score (i.e., "pred_cuisine_score"), and the list of closest "n" cuisine IDs and their corresponding cosine similarity scores (i.e., "closest_n_data").

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

 
