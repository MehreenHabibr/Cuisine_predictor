import sys
sys.path.append('..')
import pytest
import project2 as p2
import pandas as pd
import argparse
#from argparse import Namespace

from argparse import Namespace
import os
import os
import pandas as pd
import project2 
from project2 import *
import os
import pandas as pd
import project2 as p2
def test_clean_data():
    # create test data
    data = ['Chicken! Noodles.', '123  Main Street', '   garlic     ']
    
    # test function
    cleaned_data = clean_data(data)
    
    # check output types
    assert isinstance(cleaned_data, list)
    assert all(isinstance(item, str) for item in cleaned_data)
    
    # check output values
    assert len(cleaned_data) == len(data)
    assert cleaned_data[0] == 'chicken noodles'
    assert cleaned_data[1] == 'main street'
    assert cleaned_data[2] == 'garlic'
    

