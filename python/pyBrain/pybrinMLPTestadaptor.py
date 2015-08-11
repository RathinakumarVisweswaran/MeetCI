import numpy as np
import pickle
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer
from pybrain.structure import GaussianLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from optparse import OptionParser

#parsing the file name given in command line

parser=OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
(options, args)=parser.parse_args()

file_name=options.filename
print file_name

file=open(file_name)

# unpacking objects from pickle
fileObject1=open('pybrainMLP2','r')
fnn=pickle.load(fileObject1)
var_inp=fnn.indim
classes=fnn.outdim
var_out=1
fileObject=open('pybrainMLP','r')
trained_model=pickle.load(fileObject)

#Loading data
DS=ClassificationDataSet(var_inp,var_out,nb_classes=classes)
for line in file.readlines():
	data=[float(x) for x in line.strip().split(',') if x != '']
	inp=tuple(data[:var_inp])
	output=tuple(data[var_inp:])
	DS.addSample(inp,output)

_,tstdata=DS.splitWithProportion(0)	
tsdata=ClassificationDataSet(tstdata.indim,var_out,nb_classes=classes)

for i in xrange(tstdata.getLength()):
    tsdata.addSample(tstdata.getSample(i)[0],tstdata.getSample(i)[1])


tsdata._convertToOneOfMany()
testingResult=percentError(trained_model.testOnClassData(dataset=tsdata),tsdata['class'])

print "Testing accuracy: %f" % (testingResult)

