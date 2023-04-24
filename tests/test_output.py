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

import json




def test_output(capfd):
    pred = ['Indian']
    pred_cuisine_score = 0.2
    n_closest = [(1234,0.58),(4567,0.53)]
    p2.output(pred,pred_cuisine_score,n_closest)
    out,err = capfd.readouterr()
    assert type(out) == str
