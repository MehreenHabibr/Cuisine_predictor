import sys
sys.path.append('..')
import pytest
import project2
import pandas as pd
import argparse
from project2 import *

from project2 import predict_cuisine

def test_predict_cuisine():
    # create test data
    data = ['garlic chicken potatoes', 'chicken breast garlic', 'potatoes garlic', 'chicken potatoes']
    df = pd.DataFrame({'id': [1, 2, 3], 'cuisine': ['italian', 'chinese', 'mexican']})

    # test function
    args = type('', (), {'N': 2})() # create an object of args.N
    pred, pred_cuisine_score, closest_n_data = predict_cuisine(df, data, args)

    # check output types
    assert isinstance(pred, np.ndarray)
    assert isinstance(pred_cuisine_score, float)
    assert isinstance(closest_n_data, list)
    assert all(isinstance(item, tuple) for item in closest_n_data)


    assert pred_cuisine_score == 0.5
    assert len(closest_n_data) == 2

    assert closest_n_data[0][1] == 0.784


