from sacred import Experiment
import seml
import warnings

import torch

import os
import random
import numpy as np
import itertools
from sklearn import metrics

from pyts.approximation import SymbolicAggregateApproximation


from modules import GCRPlus
from modules import dataset_selecter as ds
from modules import pytorchTrain as pt
from modules import saliencyHelper as sh
from modules import helper

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sktime.classification.sklearn import RotationForest

import gc

import ViT_LRP
import cnn_LRP

from datetime import datetime

from sklearn.model_selection import StratifiedKFold

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

def arrayToString(indexes):
    out = ""
    for i in indexes:
        out = out + ',' + str(i)
    return out


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):
        if init_all:
            self.init_all()

    #init before the experiment!
    @ex.capture(prefix="init")
    def baseInit(self, useSaves, nrFolds: int, seed_value: int, symbolCount: int):
        self.seed_value = seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)# 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.RandomState(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(0)


        #save some variables for later
        self.valuesA = helper.getMapValues(symbolCount)

        self.kf = StratifiedKFold(nrFolds, shuffle=True, random_state=seed_value)
        self.kf2 = StratifiedKFold(nrFolds, shuffle=True, random_state=43)
        self.fold = 0
        self.nrFolds = nrFolds
        self.seed_value = seed_value       
        self.symbolsCount = symbolCount
        self.useSaves = useSaves


        #init gpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")


    # Load the dataset
    @ex.capture(prefix="data")
    def init_dataset(self, dataset: str, dsNumber: int, numberFeatures,  symbolicStrategy, symbolificationStrategy, doSymbolify, doFeatureExtraction):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """

        """
        - A1: auch Datentypen?
        - A4: a -1 Punkte und dafür zu e
        - A4 b und c verändert weil hier einige Probleme existierten in den vorherigen Formulierungen
        - A4 d wirkung in den Text zur Klarheit?
        - A6 b und a tauschen?
        - A7 entfernen?
        """


        self.X_train, self.X_test, self.y_train, self.y_test, self.y_trainy, self.y_testy, self.seqSize, self.dataName, self.num_of_classes, _ = ds.datasetSelector(dataset, self.seed_value, number=dsNumber) 




        self.X_trainf = []
        self.X_testf = []
        self.X_valf = []
        self.Y_trainf = []
        self.Y_trainyf = []
        self.Y_valf = []
        self.Y_testf = []
        self.Y_testyf = []

        self.X_train_orif = []
        self.X_val_orif = []
        self.X_test_orif = []

        self.columns=[]

        self.numberFeatures = numberFeatures
        self.symbolificationStrategy = symbolificationStrategy
        self.symbolicStrategy = symbolicStrategy
        self.doSymbolify =  doSymbolify
        self.doFeatureExtraction = doFeatureExtraction


        self.inDim = self.X_train.shape[1]
        self.dataset = dataset
        self.dsNumber = dsNumber

        self.dsName = str(self.dataName) +  '-n:' + str(dataset)
        print(self.dsName)



    #all inits
    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.baseInit()
        self.init_dataset()

    def printTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)


    # def trainExperiment(self, useSaves: bool, modelType: str, batch_size: int, epochs: int, numOfLayers: int, header:int, dmodel: int, dfff: int, dropout: float, att_dropout: float, doSkip: bool, doBn: bool, doClsTocken: bool, stride: int, kernal_size: int, nc: int, thresholdSet, methods): #, foldModel: int):
    # one experiment run with a certain config set. MOST OF THE IMPORTANT STUFF IS DONE HERE!!!!!!!
    @ex.capture(prefix="model")#limit:int, sizeMax:int
    def trainExperiment(self): #, foldModel: int):

        print('Dataname:')
        print(self.dsName)
        self.printTime()
        warnings.filterwarnings('ignore')   

        fullResults = dict()
        
        fullResultDir = 'preprocessingSymbolicGCR' 


        wname = pt.getDatasetName(self.dataset, self.dsNumber, self.numberFeatures,  self.symbolicStrategy, self.symbolificationStrategy, self.doSymbolify, self.doFeatureExtraction, self.symbolsCount, self.nrFolds, self.seed_value, resultsPath = fullResultDir)
        if os.path.isfile(wname + '.pkl'):

            fullResults["Error"] = "dataset " + self.dsName + " already done: " + str(self.seqSize) + "; name: " + wname
            print('Already Done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname)
        
            return "dataset " + self.dataName + "already done: " + str(self.seqSize)  + "; name: " + wname 




        doSplit = True
        try:
            for train, test in self.kf.split(self.X_train, self.y_trainy):
                print('ok')
        except:
            print("5 fold split fail on " + self.dataName)
            doSplit = False

        self.fold = 0
        if doSplit:
            for train, test in self.kf.split(self.X_train, self.y_trainy):

                self.fold+=1
                print(f"Fold #{self.fold}")
                
                
                x_train1 = self.X_train[train]
                x_val = self.X_train[test]
                y_train1 = self.y_train[train]
                y_trainy1 = self.y_trainy[train]
                y_val = self.y_train[test]
                
                
                if self.symbolificationStrategy == 'uniform':
                    symbolicMethod2 = 'sax'
                else:
                    symbolicMethod2 = 'mcb'
                x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy1, y_testy, top_n_features = helper.preprocessDataWithFeatures(x_train1, x_val, self.X_test, y_train1, y_val, self.y_test, y_trainy1, self.y_testy, self.fold, self.symbolsCount, self.dataName, doSymbolify = self.doSymbolify, doFeatureExtraction=self.doFeatureExtraction, n=self.numberFeatures, symbolicMethod=self.symbolicStrategy, symbolicMethod2=symbolicMethod2, symbolicStrategy2=self.symbolificationStrategy, useSaves = self.useSaves, doSelect=True, doDataAugmentation=False, augmentationNumer=1, doOrdinalPatterns=False, doConcatinate=False)

                for t in np.unique(x_train1):
                    if t not in self.valuesA:
                        print(t)
                        print(self.valuesA)
                        a = 1
                        print(a + '1')
                for t in np.unique(x_test):
                    if t not in self.valuesA:
                        print(t)
                        print(self.valuesA)
                        a = 1
                        print(a + '1')
                self.X_trainf.append(x_train1)
                self.X_testf.append(x_test)
                self.X_valf.append(x_val)
                self.Y_trainf.append(y_train1)
                self.Y_trainyf.append(y_trainy1)
                self.Y_valf.append(y_val)
                self.Y_testf.append(y_test)
                self.Y_testyf.append(y_testy)

                self.X_train_orif.append(X_train_ori)
                self.X_val_orif.append(X_val_ori)
                self.X_test_orif.append(X_test_ori)

                self.columns.append(top_n_features)
        else: 
            x_train1 = self.X_train#[train]
            x_val = self.X_train#[test]
            y_train1 = self.y_train#[train]
            y_trainy1 = self.y_trainy#[train]
            y_val = self.y_train#[test]
            
            
            if self.symbolificationStrategy == 'uniform':
                symbolicMethod2 = 'sax'
            else:
                symbolicMethod2 = 'mcb'
            x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy1, y_testy, top_n_features = helper.preprocessDataWithFeatures(x_train1, x_val, self.X_test, y_train1, y_val, self.y_test, y_trainy1, self.y_testy, self.fold, self.symbolsCount, self.dataName, doSymbolify = self.doSymbolify, doFeatureExtraction=self.doFeatureExtraction, n=self.numberFeatures, symbolicMethod=self.symbolicStrategy, symbolicMethod2=symbolicMethod2, symbolicStrategy2=self.symbolificationStrategy, useSaves = self.useSaves, doSelect=True, doDataAugmentation=False, augmentationNumer=1, doOrdinalPatterns=False, doConcatinate=False)
            for fl in range(self.nrFolds):

                for t in np.unique(x_train1):
                    if t not in self.valuesA:
                        print(t)
                        print(self.valuesA)
                        a = 1
                        print(a + '1')
                for t in np.unique(x_test):
                    if t not in self.valuesA:
                        print(t)
                        print(self.valuesA)
                        a = 1
                        print(a + '1')

                self.X_trainf.append(x_train1)
                self.X_testf.append(x_test)
                self.X_valf.append(x_val)
                self.Y_trainf.append(y_train1)
                self.Y_trainyf.append(y_trainy1)
                self.Y_valf.append(y_val)
                self.Y_testf.append(y_test)
                self.Y_testyf.append(y_testy)

                self.X_train_orif.append(X_train_ori)
                self.X_val_orif.append(X_val_ori)
                self.X_test_orif.append(X_test_ori)
                self.columns.append(top_n_features)




        fullResults['X_trainf'] = self.X_trainf
        fullResults['X_testf'] = self.X_testf
        fullResults['X_valf'] = self.X_valf
        fullResults['Y_trainf'] = self.Y_trainf
        fullResults['Y_valf'] = self.Y_valf
        fullResults['Y_testf'] = self.Y_testf

        fullResults['X_train_orif'] = self.X_train_orif
        fullResults['X_val_orif'] = self.X_val_orif
        fullResults['X_test_orif'] = self.X_test_orif
        
        fullResults['columns'] = self.columns

        
        saveName = pt.getDatasetName(self.dataset, self.dsNumber, self.numberFeatures,  self.symbolicStrategy, self.symbolificationStrategy, self.doSymbolify, self.doFeatureExtraction, self.symbolsCount, self.nrFolds, self.seed_value, resultsPath = fullResultDir)

        print(saveName)
        helper.save_obj(fullResults, str(saveName))

        self.printTime()

        return saveName


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.trainExperiment()
