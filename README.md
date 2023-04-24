**MEHREEN HABIB**
---------
> ## Project Title: CS5293, spring 2023 Project 2
### Project Description
 **This project aims to predict the type of cuisine based on a list of ingredients entered through the command line interface (CLI). Additionally, it will provide a predicted score for the given cuisine and a list of the top "n" cuisines that are most similar to the input ingredients.**
 
 Command for running the project:
> pipenv run python project2.py --N 5 --ingredient paprika --ingredient "black olives", --ingredient  "grape tomatoes", --ingredient "garlic", --ingredient "pepper", --ingredient "purple onion", --ingredient "seasoning", --ingredient "garbanzo beans", --ingredient "feta cheese crumbles"

Command for running the test scripts:
> pipenv run python -m pytest

 **Modules**
 1. argparse : This provides a convenient way to parse command line arguments in Python.
 2. json : This  provides functions for working with JSON data 
 3. numpy as np : This provides support for working with arrays and numerical operations
 4. pandas as pd : This provides data structures and functions for working with tabular data
 5. from sklearn.metrics.pairwise import cosine_similarity : This line of code imports the "cosine_similarity" function from the "sklearn.metrics.pairwise" module, which provides support for calculating pairwise similarity between vectors using the cosine similarity measure.
 6. from sklearn.model_selection import train_test_split : This line of code imports the "train_test_split" function from the "sklearn.model_selection" module, which provides support for splitting data into training and testing sets for machine learning models.
 7. from sklearn.feature_extraction.text import TfidfVectorizer : This line of code imports the "TfidfVectorizer" class from the "sklearn.feature_extraction.text" module, which provides support for extracting text features using the Term Frequency-Inverse Document Frequency (TF-IDF) weighting scheme.
 8. from sklearn.neighbors import KNeighborsClassifier : This line of code imports the "KNeighborsClassifier" class from the "sklearn.neighbors" module, which provides support for building K-Nearest Neighbors (K-NN) classification models.



 #### Approach to Develope the code
---
1. `read_file()`
The "read_file" function reads the contents of a JSON file named "yummly.json" using a context manager, converts the JSON data into a Pandas DataFrame, and returns the resulting DataFrame.
2. `clean_data(data)`
   The "clean_data" function takes a list of strings as input and performs the following operations:

   - Converts all strings to lowercase.
   - Removes all punctuation marks from the strings.
   - Removes all numerical characters from the strings.
   - Removes any extra whitespaces from the beginning or end of the strings.
   - The function then returns the cleaned list of strings.
3. `vectorize_form(df, args)`
   The "vectorize_form" function take input ingredients, cleans the data by converting to lowercase, removing punctuation and numerical characters, and joining the input ingredients with the cleaned data. It returns the cleaned list of strings.
4. `predict_cuisine(df, data, args)`
   The function takes a dataframe, a list of data, and some arguments as input. It then uses a TfidfVectorizer and KNeighborsClassifier to predict the cuisine label of the input data and returns the predicted label, the corresponding score, and the top "n" closest matches. 
   It also creates a dictionary of the redacted named entities and their labels and returns the redacted text, the dictionary, and the count of named entities that were redacted.
5. `output(pred, pred_cuisine_score, n_closest)`
    This function defines a dictionary object with three key-value pairs: "cuisine", "score", and "closest". It then prints the dictionary in JSON format with an indentation level of 4.
6.  `main(arser)`
    The main function takes command-line arguments, reads a file, vectorizes its data, predicts the cuisine based on input ingredients, and outputs the result. If the script is run as the main program, it uses argparse to parse the command-line arguments.
 #### Tests
---
1. test_redact_names.py :  Test if names are redacted correctly
2. test_redact_gender.py : Test if genders are redacted correctly
3. test_redact_address.py : Test if addresses are redacted correctly
4. test_redact_dates.py : Test if dates are redacted correctly
5. test_redact_output.py :  Test if the output is written to a file.
6. test_redact_phones.py : Test if phone numbers are redacted correctly
7. test_unicode.py : Test if the unicode character is replaced correctly

#### Pytests can run using below command
`pipenv run python -m pytest`

pipenv install
Packages required to run this project are kept in requirements.txt file which automatically installs during installation of pipenv in step 1.


##### python_version = "3.10"

##### pytest==7.2.2

Once, the packages are successfully installed

#### The project can be executed using
`pipenv run python redactor.py --input *.txt --names --dates --phones --genders --address --output 'files' --stats stdout`



## Assumptions:
---
1. Names of people and organizations are considered as names and thus redacted if --names flag is used.
2. Accuracy and performance of this application is directly dependent on SpaCy model.
3. This tools accuracy and performance is enhanced by using regular expressions along with SpaCy but unfortunately not all cases of the entities (names, phones, genders, dates and addresses) were included as regular expressions. Thus, some information may not be redacted if they were not recognized by SpaCy model or included regular expressions.
4. Address is redacting zip code sometime and sometime it is not.


![](https://github.com/MehreenHabibr/cs5293sp23-project1/blob/main/Recording%20%236.gif)
