
import pandas as pd
import numpy as np
from optparse import OptionParser
from format import prep

#--------------------------------------------------------#
# process inputs
#--------------------------------------------------------#

parser = OptionParser()
parser.add_option("-o", action="store", 
                  type="int", default=-1, dest="opt")
parser.add_option("-f", action="store",
                  type="string", default="data/train.csv",
                  dest="filename")                  
parser.add_option("--save", action="store_true",
                  default = False, dest = "save")

# Load inputs
options, args = parser.parse_args()

# Check inputs
opt = options.opt
if opt == -1:
    print "Choose valid option"
    import sys
    sys.exit()

f_data = options.filename
eval = False
if 'test' in f_data:
    eval = True

wesave = options.save

#--------------------------------------------------------#
# Prepare data
#--------------------------------------------------------#

data = pd.read_csv(f_data)
data, keeps = prep(data, eval)
import sys
sys.exit()
#--------------------------------------------------------#
# Choose a method
#--------------------------------------------------------#

# K-nn training and testing
if opt == 0:
    from KNN import KNN_test_train
    KNN_test_train(data[ keeps[:-1] ], data[ keeps[-1] ], wesave)

# BDT
if opt == 1:
    from BDT import BDT_test_train
    BDT_test_train(data[ keeps[:-1] ], data[ keeps[-1] ])

# Save K-nn
if opt == 2:
    from KNN import KNN_save
    KNN_save( data[ keeps[:-1] ], data[ keeps[-1] ] )

# Write K-nn output
if opt == 3:
    from KNN import KNN_evaluate
    KNN_evaluate(data)
