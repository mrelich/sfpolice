
import pandas as pd
import numpy as np
from optparse import OptionParser
from format import prep, categories, districts
import pickle

from KNN import KNN_evaluate

#--------------------------------------------------------#
# process inputs
#--------------------------------------------------------#

parser = OptionParser()
parser.add_option("-o", action="store", 
                  type="int", default=-1, dest="opt")

# Load inputs
options, args = parser.parse_args()

# Check inputs
opt = options.opt
if opt == -1:
    print "Choose valid option"
    import sys
    sys.exit()

#--------------------------------------------------------#
# Prepare data differently.  Load in chuncks since it 
# takes so much memory to process
#--------------------------------------------------------#

f_data = "data/test.csv"
data = pd.read_csv(f_data, chunksize=50000)

# Loop over each chunk and run options
probs = None
counter = 0
for chunk in data:
    print "Working on ", counter, "..."
    chunk, keeps = prep(chunk, True)
    
    
    # Run KNN and get probs
    if opt == 0:
        if probs == None:
            probs = KNN_evaluate(chunk[keeps])
        else:
            probs = np.concatenate([probs, KNN_evaluate(chunk[keeps])])


    counter += 1

# Now we are done, write the output
outcat = np.array(np.arange(len(categories)),dtype='string')
for key in categories:
    outcat[categories[key]] = key

output = pd.DataFrame(probs,columns=outcat)

outname = ""
if opt == 0: outname = "submission_knn.csv"
else:        
    print "wtf pick an option"
    import sys
    sys.exit()

output.to_csv(outname)
