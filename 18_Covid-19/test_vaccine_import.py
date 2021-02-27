import os.path as path
import sys
from inspect import getsourcefile

# setup absolute path
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from vaccine_import import *

# test data vaccine_import
def test_total_confirmed_vaccine_global_import():
    assert total_confirmed_vaccine_global.size > 0

def test_total_confirmed_vaccine_us_import():
    assert total_confirmed_vaccine_us.size > 0
