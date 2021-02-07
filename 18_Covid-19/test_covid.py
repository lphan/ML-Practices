import os.path as path
import sys
from inspect import getsourcefile

# setup absolute path
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

from covid import *

# TODO: test data covid

# totalconfirmed_by_day > totalrecovered_by_day, totalconfirmed_by_day > totalfatalities_by_day

# np.array(ratioFatalByDay) < np.array(ratioRecByDay)

# us recovered cases

# tbd.
