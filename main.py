import numpy as np
import pickle
import  xml.etree.ElementTree as ET
import meetCI as ml
from sets import Set
from optparse import OptionParser
import os
import subprocess

#parsing the XML file

parser=OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
(options, args)=parser.parse_args()

file_name=options.filename

#file_name='/home/suhaspillai/ExpertSystem_MLSoftware/XMLfiles/RBF1.xml'
#rootObj=ml.parse('RBF1.xml')
tree=ET.parse(file_name)
root=tree.getroot()

#need to figure out how to parse the xml and then call other stuff
s_MultiLayerPerceptron=Set(['Fann','pyBrain','theano','Neuroph'])
s_RadialBasisFunctionNetwork=Set(['pyBrain','Neuroph'])
s_RecurrentNeuralNetworks=Set(['pyBrain'])
s_SupportVectorMachines=Set(['Scikit'])
s_RandomForest=Set(['Scikit'])
'''
d_theano={'MultiLayerPerceptron'}
d_pyBrain={'MultiLayerPerceptron','RadialBasisFunctionNetwork','RecurrentNeuralNetwork'}
d_Fann={'MultiLayerPerceptron'}
d_scikit={'SupportVectorMachines','RandomForest'}
'''
dict_lib={}
dict_lib['classification']=[s_MultiLayerPerceptron | s_RadialBasisFunctionNetwork | s_SupportVectorMachines | s_RandomForest]
dict_lib['prediction']=[s_RecurrentNeuralNetworks]
dict_lib['expertsystem']=Set(['Jess'])

dict_algorithm={}
dict_algorithm['classification']=['MultiLayerPerceptron','RadialBasisFunctionNetwork','SupportVectorMachines','RandomForest']
dict_algorithm['prediction']=['RecurrentNeuralNetwork']
dict_algorithm['expertsystem']=['Jess']              

#dict_all['ExpertSystems']=['Jess']

AI_type=root.find('MachineLearning')

if AI_type is None:
    print 'expertsystem'
    
else:
    
    for child in AI_type:
        problem_type=child.tag


    if problem_type=="classification":
        algorithm=root.find("MachineLearning/classification/algorithm")

        for child in algorithm:
            algorithm_name=child.tag

        if algorithm_name in dict_algorithm['classification']:
            print dict_lib['classification']
            print '\n'
            response=raw_input('Please type the name of the library which you would like to use for execution: \n')
            if response=='pyBrain':
                from python.pyBrain.PyBrain_RBF_Iris import exec_algo
                exec_algo()
            elif response=='Neuroph':
                 p=os.popen('java -jar Java/Neuroph/JavaPlugin.jar Java/Neuroph/JavaPlugin/examples/MLP_Iris.xml Java/Neuroph/JavaPlugin/output','r')
            elif response=='Fann':
                 subprocess.call('./C/FANN-2.2.0-Source/examples/FANN_MLP_Iris')
             
    elif problem_type=="prediction":
        print 'hi'
        algorithm=root.find("MachineLearning/prediction/algorithm")
        for child in algorithm:
            algorithm_name=child.tag
            print algorithm_name
        if algorithm_name in dict_algorithm['prediction']:
            print dict_lib['prediction']
            print '\n'
            response=raw_input('Please type the name of the library which you would like to use for execution: \n')
            if response=='pyBrain':
                from python.pyBrain.RNN import exec_algo
                exec_algo()
    
     
             
              
print 'Execution completed'          


'''
#calling java stuff
            
            import os
             p=os.popen('java -jar JavaPlugin.jar JavaPlugin/examples/MLP_Iris.xml JavaPlugins/output','r')
             
            while 1:
	line=p.readline()
	if not line: break
	print line
            
            
import subprocess
subprocess.call('./FANN_MLP_Iris')    
'''        
        

        
    
    
    


