from sacred import Experiment
import seml
import warnings

import torch

import os
import random
import numpy as np
#from pynvml import *
import itertools
from sklearn import metrics

from pyts.approximation import SymbolicAggregateApproximation


#from modules import transformer
from modules import GCRPlus
from modules import dataset_selecter as ds
from modules import pytorchTrain as pt
from modules import saliencyHelper as sh
from modules import helper
import psutil

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


ram = psutil.virtual_memory()

def printCheck(path, msg):
    f = open(path+ "notes.txt", "a")
    f.write(msg +'\n')
    f.write("RAM usage (%):"+ str(ram.percent) + '\n')
    f.write("RAM used (GB):"+ str(round(ram.used / 1e9, 2)) + '\n')
    f.close()


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
    def baseInit(self, nrFolds: int, patience: int, seed_value: int, symbolCount: int, useSaves: bool):
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
        self.patience = patience
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

        """
        

        doSplit = True
        try:
            for train, test in self.kf.split(self.X_train, self.y_trainy):
                print('ok')
        except:
            print("5 fold split fail on " + self.dataName)
            doSplit = False

        """
        _, _, _, _, _, _, self.seqSize, self.dataName, self.num_of_classes, _ = ds.datasetSelector(dataset, self.seed_value, number=dsNumber) 

        """
        self.sizeMax = 2000
        if len(self.xtrain) > self.sizeMax or len(self.xtest) > self.sizeMax:
            #fullResults["Error"] = "dataset " + self.dataName + " to big: " + str(self.seqSize)
            print('TO LONGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " to big: " + str(self.seqSize))
            
            ##print(1 + 'b')
            return "dataset " + self.dataName + " to big: " + str(self.seqSize)
        else:
            print(1 + 'b')
        """


        self.numberFeatures = numberFeatures
        self.symbolificationStrategy = symbolificationStrategy
        self.symbolicStrategy = symbolicStrategy
        self.doSymbolify =  doSymbolify
        self.doFeatureExtraction = doFeatureExtraction

        wname = pt.getDatasetName(dataset, dsNumber, self.numberFeatures,  self.symbolicStrategy, self.symbolificationStrategy, self.doSymbolify, self.doFeatureExtraction, self.symbolsCount, self.nrFolds, self.seed_value, resultsPath = "preprocessingSymbolicGCR2")
        if os.path.isfile(wname + '.pkl'):
            data = helper.load_obj(str(wname))

            for index, v in np.ndenumerate(data):
                fullData = v

            
        else:
            print('Preprocessing not found ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset preprocessing not found; name: " + wname)
            b = 2
            print(b + 'b')
        
            return "dataset preprocessing not found name: " + wname 


        self.datasetName = wname + '.pkl'
        self.X_trainf = fullData['X_trainf']
        self.X_testf = fullData['X_testf'] 
        self.X_valf =fullData['X_valf'] 
        self.Y_trainf =fullData['Y_trainf']
        #self.Y_trainyf = np.argmax(self.Y_trainf, axis=1)+1 ##fullData['Y_trainyf'] 
        self.Y_valf =fullData['Y_valf']
        self.Y_testf =fullData['Y_testf'] 
        #print(np.array(self.Y_testf).shape)
        #print(self.Y_testf[0][0])
        #self.Y_testyf = np.argmax(np.array(self.Y_testf), axis=1)+1

        self.X_train_orif = fullData['X_train_orif']
        self.X_val_orif = fullData['X_val_orif']
        self.X_test_orif = fullData['X_test_orif'] 

        self.fold = 0
        """
        preprocessingFolder = 'preprocessingSymbolicGCR'
        wname = pt.getDatasetName(self.dataset, self.dsNumber, self.numberFeatures,  self.symbolicStrategy, self.symbolificationStrategy, self.doSymbolify, self.doFeatureExtraction, self.symbolsCount, self.nrFolds, self.seed_value, resultsPath = preprocessingFolder)

        """
        
        self.dataset = dataset
        self.dsNumber = dsNumber
        self.dataName = ds.univariate_equal_length[dsNumber]
        self.dsName = str(self.dataName) +  '-n:' + str(dataset)
        print(self.dsName)

        self.y_test = self.Y_testf[0]
        if len(self.Y_testf) == 5:
            self.y_testy = np.argmax(self.Y_testf[0], axis=1)
        else:
            self.y_testy = self.Y_testf[1]



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
    def trainExperiment(self, hypers, batch_size: int, stride: int, kernal_size: int, nc: int, thresholdSteps, methods): #, foldModel: int):


        """
        if len(self.xtrain) > self.sizeMax or len(self.xtest) > self.sizeMax:
            #fullResults["Error"] = "dataset " + self.dataName + " to big: " + str(self.seqSize)
            print('TO LONGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " to big: " + str(self.seqSize))
            
            ##print(1 + 'b')
            return "dataset " + self.dataName + " to big: " + str(self.seqSize)
        else:
            print(1 + 'b')
        """


        print('Dataname:')
        print(self.dsName)
        self.printTime()
        warnings.filterwarnings('ignore')   

        fullResults = dict()
        
        modelType = hypers[0]
        epochs = hypers[1]
        dmodel = hypers[2]
        dfff = hypers[3]
        doSkip = hypers[4]
        doBn = hypers[5]
        header = hypers[6]
        numOfLayers = hypers[7]
        dropout = hypers[8]
        att_dropout = hypers[9]
        if modelType == 'Transformer':
            doClsTocken = hypers[10]
        else:
            doClsTocken = False

        fullResultDir = 'pResultSymbolicGCR2' 
        filteredResults= 'filteredSymbolicGCR3'



        


        #symbolificationStrategy, symbolCount, numberFeatures
        wname = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, self.symbolsCount, self.symbolicStrategy ,self.symbolificationStrategy, self.numberFeatures, self.doSymbolify, self.doFeatureExtraction, learning = False, results = True, resultsPath=filteredResults)
        if os.path.isfile(wname + '.pkl'):

            fullResults["Error"] = "dataset " + self.dsName + " already done: " + str(self.seqSize) + "; name: " + wname
            print('Already Done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname)
        
            return "dataset " + self.dataName + "already done: " + str(self.seqSize)  + "; name: " + wname 

    

        #print(1 + 'b')

        doTraining = True
        self.printName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, self.symbolsCount, self.symbolicStrategy ,self.symbolificationStrategy, self.numberFeatures, self.doSymbolify, self.doFeatureExtraction, learning = False, results = True, resultsPath='msges')
        printCheck(self.printName, 'Start')


        wname = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, self.symbolsCount, self.symbolicStrategy ,self.symbolificationStrategy, self.numberFeatures, self.doSymbolify, self.doFeatureExtraction, learning = False, results = True, resultsPath=fullResultDir)

        if os.path.isfile(wname + '.pkl'):
            print('Already Done training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("trained dataset " + self.dataName + " already done: " + str(self.seqSize) + "; name: " + wname)
            try:
                results = helper.load_obj(str(wname))#, bigLoad=True)

                for index, v in np.ndenumerate(results):
                    fullResults = v

                doTraining = False
                    
            except:
                print("Error on loading file! Removing file!")
                print(wname + '.pkl')
                #os.remove(wname + '.pkl') 
                #a = 1
                #print(a + '1') 
                #return "Ignore"

        printCheck(self.printName, 'Do training: ' +str(doTraining))

        if doTraining:
            #fullResults['testData'] = self.X_test
            fullResults['testTarget'] = self.Y_testf[0]

            #save all params!
            #TODO neue params adden und alte löschen!
            fullResults['params'] = dict()
            fullResults['params']['symbols'] = self.symbolsCount
            fullResults['params']['patience'] = self.patience
            fullResults['params']['fileName'] = self.dsName
            fullResults['params']['epochs'] = epochs
            fullResults['params']['batchSize'] = batch_size
            fullResults['params']['useSaves'] = self.useSaves
            fullResults['params']['numOfLayers'] = numOfLayers
            fullResults['params']['header'] = header
            fullResults['params']['dmodel'] = dmodel
            fullResults['params']['dfff'] = dfff
            fullResults['params']['dropout'] = dropout
            fullResults['params']['att_dropout'] = att_dropout
            fullResults['params']['doSkip'] = doSkip
            fullResults['params']['doBn'] = doBn 
            fullResults['params']['modelType'] = modelType
            fullResults['params']['doClsTocken'] = doClsTocken
            fullResults['params']['dataset'] = self.dataset
            fullResults['params']['dsNumber'] = self.dsNumber
            print(fullResults['params'])

            fullResults['results'] = dict()
            resultDict = fullResults['results']
            resultDict['trainPred'] = []
            resultDict['trainAcc'] = []
            resultDict['trainLoss'] = []

            resultDict['valPred'] = []
            resultDict['valAcc'] = []
            resultDict['valLoss'] = []

            resultDict['testPred'] = []
            resultDict['testAcc'] = []
            resultDict['testLoss'] = []

            resultDict['trainData'] = []
            resultDict['testData'] = []
            resultDict['trainTarget'] = []
            resultDict['valData'] = []
            resultDict['valTarget'] = []

            resultDict['treeScores'] = []
            resultDict['treeImportances'] = []

            resultDict['assembly acc'] = 0
            resultDict['assembly rec'] = 0
            resultDict['assembly prec'] = 0
            resultDict['assembly f1'] = 0
            for fold in range(len(self.X_trainf)):

                x_train1 = self.X_trainf[fold]
                x_test = self.X_testf[fold]
                x_val = self.X_valf[fold]
                y_train1 = self.Y_trainf[fold]
                y_trainy1 = np.argmax(self.Y_trainf[fold],axis=1)
                y_val = self.Y_valf[fold]
                if len(self.Y_testf) == 5:
                    y_test = self.Y_testf[fold]
                    y_testy = np.argmax(self.Y_testf[fold], axis=1)
                else:
                    y_test = self.Y_testf[fold*2]
                    y_testy = self.Y_testf[(fold*2)+1]


                self.inDim = self.X_testf[fold].shape[1]

                for t in np.unique(x_train1):
                    if t not in self.valuesA:
                        print(t)
                        print(self.valuesA)
                        print(self.datasetName)
                        a = 1
                        #os.remove(self.datasetName)
                        print(a + '1')
                for t in np.unique(x_test):
                    if t not in self.valuesA:
                        print(t)
                        print(self.valuesA)
                        print(self.datasetName)
                        a = 1
                        #os.remove(self.datasetName)
                        print(a + '1')

                #x_test = self.X_test.copy()

                if modelType == 'CNN':
                    #model = cnn_LRP.ResNetLikeClassifier(outputs=self.num_of_classes, inDim=self.inDim, num_hidden_layers=2, nc=1, nf=8, dropout=0.3, maskValue = -2, stride=1, kernel_size=3, doSkip=True, doBn=True)
                    model = cnn_LRP.ResNetLikeClassifier(outputs=self.num_of_classes, inDim=self.inDim, num_hidden_layers=numOfLayers, nc=nc, nf=dmodel, dropout=dropout, maskValue = -2, stride=stride, kernel_size=kernal_size, doSkip=doSkip, doBn=doBn)
                    
                    if True:
                        x_train1 = np.expand_dims(x_train1,1)
                        x_val = np.expand_dims(x_val,1)
                        x_test = np.expand_dims(x_test,1)
                elif modelType == 'Transformer':

                    model = ViT_LRP.TSModel(num_hidden_layers=numOfLayers, inDim=self.inDim, dmodel=dmodel, dfff=dfff, num_heads=header, num_classes=self.num_of_classes, dropout=dropout, att_dropout=att_dropout, doClsTocken=doClsTocken)
                else:  
                    raise ValueError('Not a valid model type: ' + modelType)

                print('Train data shapes:')
                print(x_train1.shape)
                print(y_train1.shape)
                
                print(x_val.shape)
                print(y_val.shape)
                print(x_test.shape)
                print(y_test.shape)

                if modelType == 'Tree':
                    model, trainPred, trainAcc, trainLoss, valPred, valAcc, valLoss, testPred, testAcc, testLoss = pt.trainTree(model, x_train1, y_train1, x_val, y_val, x_test, y_test)
                else:
                    model.double()
                    model.to(self.device)
                    model, trainPred, trainAcc, trainLoss, valPred, valAcc, valLoss, testPred, testAcc, testLoss = pt.trainBig(self.device, model, x_train1, y_train1, x_val, y_val, x_test, self.patience, self.useSaves, y_test, batch_size, epochs, fileAdd=self.dsName)
                    model.eval()

                torch.cuda.empty_cache()

                train_predictions = y_trainy1 #np.argmax(y_train1, axis=1)+1
                test_predictions = y_testy #np.argmax(self.y_test, axis=1)+1

                newT = np.squeeze(x_train1)
                newG = np.squeeze(x_test)
                clf = RandomForestClassifier()
                clf.fit(newT, train_predictions)
                scores = clf.score(newG, test_predictions)

                #TODO brauch ich wirklich die train data? Lieber irgendwo einzeln abspeichern? Inc format etc -> fürs erste ja
                #resultDict['treeScores'].append(scores)
                #resultDict['treeImportances'].append(clf.feature_importances_)
                
                resultDict['trainPred'].append(trainPred) 
                #resultDict['trainAcc'].append(trainAcc)
                #resultDict['trainLoss'].append(trainLoss)

                #resultDict['valPred'].append(valPred)
                #resultDict['valAcc'].append(valAcc)
                #resultDict['valLoss'].append(valLoss)

                resultDict['testPred'].append(testPred)
                #resultDict['testAcc'].append(testAcc)
                #resultDict['testLoss'].append(testLoss)

                resultDict['trainData'].append(x_train1)
                resultDict['testData'].append(x_test)
                resultDict['trainTarget'].append(y_train1)
                #resultDict['valData'].append(x_val)
                #resultDict['valTarget'].append(y_val)

                if 'saliency' not in fullResults.keys():
                    fullResults['saliency'] = dict()
                saliencies = fullResults['saliency']

                for method in methods[modelType].keys():
                    for submethod in methods[modelType][method]:
                        if method+'-'+submethod not in saliencies.keys():
                            saliencies[method+'-'+submethod] = dict()
                            outMap = saliencies[method+'-'+submethod]
                            outMap['Fidelity'] = dict()
                            outMap['Infidelity'] = dict()
                            outMap['outTrain'] = []
                            outMap['outVal'] = []
                            outMap['outTest'] = []
                            outMap['modelTrain'] = []
                            outMap['modelVal'] = []
                            outMap['modelTest'] = []
                            outMap['means'] = dict()
                            outMap['means']['outTrain'] = []
                            outMap['means']['outVal'] = []
                            outMap['means']['outTest'] = []
                            outMap['means']['modelTrain'] = []
                            outMap['means']['modelVal'] = []
                            outMap['means']['modelTest'] = []
                            outMap['classes'] = dict()
                            for c in range(self.num_of_classes):
                                outMap['classes'][str(c)] = dict()
                                outMap['classes'][str(c)]['outTrain']= []
                                outMap['classes'][str(c)]['outVal'] = []
                                outMap['classes'][str(c)]['outTest'] = []
                            outMap['TargetClasses'] = dict()
                            outMap['ModelClasses'] = dict()
                        outMap = saliencies[method+'-'+submethod]
                        if submethod.startswith('smooth'):
                            smooth = True
                        else:
                            smooth = False
                        
                        
                        model.eval()
                        self.printTime()
                        printCheck(self.printName, method+'-'+submethod + 'start' )
                        _, _, _, _, _ = sh.getSaliencyMap(outMap, "out", self.device, self.num_of_classes, modelType, method, submethod, model, x_train1, x_val, x_test, trainPred, valPred, testPred, smooth, doClassBased=True)
                        printCheck(self.printName, method+'-'+submethod + 'end')
                        #sh.mapSaliency(outMap['ModelClasses'], self.num_of_classes, outTrain, trainPred, outVal, valPred, outTest, testPred, do3DData=data3D)
                        self.printTime()
                        gc.collect()
                        torch.cuda.empty_cache()

            fullResults['assembly predictions'] = np.mean(np.array(resultDict['testPred']), axis=0)
            testPredy = np.argmax(np.mean(np.array(resultDict['testPred']), axis=0), axis=1)
            
            fullResults['assembly acc'] = metrics.accuracy_score(testPredy, np.argmax(self.y_test,axis=1))
            fullResults['assembly rec'] = metrics.recall_score(testPredy, np.argmax(self.y_test,axis=1), average='macro')
            fullResults['assembly prec'] = metrics.precision_score(testPredy, np.argmax(self.y_test,axis=1), average='macro')
            fullResults['assembly f1'] = metrics.f1_score(testPredy, np.argmax(self.y_test,axis=1), average='macro')


        if doTraining:
            printCheck(self.printName, 'assembly acc:'  + str(fullResults['assembly acc']))
            printCheck(self.printName, 'assembly acc:'  + str(fullResults['assembly acc']))
            for s in fullResults['results']['testPred']:
                testPredy = np.argmax(np.mean(np.array(fullResults['results']['testPred']), axis=0), axis=1)
                printCheck(self.printName,'fold model acc' + str(metrics.accuracy_score(np.argmax(s, axis=1), testPredy)))
            #saveName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, self.symbolsCount, self.symbolicStrategy ,self.symbolificationStrategy, self.numberFeatures, self.doSymbolify, self.doFeatureExtraction, learning = False, results = True, resultsPath=fullResultDir)
            #print(saveName)
            #try:
            #helper.save_obj(fullResults, str(saveName))#, bigLoad=True)
        saveName = self.evaluateAndSaveResults(fullResults, self.useSaves, hypers, batch_size, stride, kernal_size, nc, methods, filteredResults)
        #except:
            
            #
            #a = 1
            #print(a + '3')

        #saveName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, self.symbolsCount, self.symbolicStrategy ,self.symbolificationStrategy, self.numberFeatures, self.doSymbolify, self.doFeatureExtraction, learning = False, results = True, resultsPath=fullResultDir)

        #print(saveName)
        #helper.save_obj(fullResults, str(saveName))

        self.printTime()

        return saveName


    def evaluateAndSaveResults(self, res, useSaves: bool, hypers, batch_size: int, stride: int, kernal_size: int, nc: int, methods, filteredResults):
        self.printTime()
        modelType = hypers[0]
        epochs = hypers[1]
        dmodel = hypers[2]
        dfff = hypers[3]
        doSkip = hypers[4]
        doBn = hypers[5]
        header = hypers[6]
        numOfLayers = hypers[7]
        dropout = hypers[8]
        att_dropout = hypers[9]
        if modelType == 'Transformer':
            doClsTocken = hypers[10]
        else:
            doClsTocken = False

        fullResults = res

        """
        print('Dataname:')
        print(self.dsName)
        
        warnings.filterwarnings('ignore')   

        wname = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True, resultsPath='filteredResults')
        if os.path.isfile(wname + '.pkl'):
            fullResults["Error"] = "dataset " + self.dsName + " already done: " + wname
            print('Already Done ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("dataset " + self.dataName + " already done: " + wname)
        
            return "dataset " + self.dataName + "already done: " + wname #fullResults

        saveName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, dropout, att_dropout, doSkip, doBn, doClsTocken, learning = False, results = True)
        #os.remove(str(saveName)) 

        results = helper.load_obj(str(saveName))

        res = dict()
        for index, v in np.ndenumerate(results):
            res = v
        """
        
        #trueIndexes = res['params']['trueIndexes']
        nrSymbols = res['params']['symbols']
        symbolA = helper.getMapValues(nrSymbols)
        symbolA = np.array(symbolA)
        #trueSymbols = symbolA[trueIndexes]
        #falseSymbols = np.delete(symbolA, trueIndexes)

        modes = [0]
        modesMap = { 'LRP-rollout' : 0, 'captum-IntegratedGradients': 2, 'captum-FeaturePermutation': 1, 'Attention-.': 0, 'Random-Random': 0} 

        tl = res['testTarget'] 
        #gt = res['testData']
        num_of_classes = self.num_of_classes #len(set(list(tl.flatten())))
        
        finalResults = dict()
        finalResults['model acc'] = []
        finalResults['model prec'] = []
        finalResults['model rec'] = []
        finalResults['model f1'] = []

        finalResults['simpleGCR'] = dict()
        finalResults['simpleGCR']['rMA'] = dict()
        finalResults['simpleGCR']['rMS'] = dict()

        finalResults['assembly predictions'] = res['assembly predictions'] 
        finalResults['assembly acc'] = res['assembly acc']
        finalResults['assembly rec'] = res['assembly rec'] 
        finalResults['assembly prec'] = res['assembly prec']
        finalResults['assembly f1'] = res['assembly f1']

        for normName in GCRPlus.getAllNeededReductionNames():
            finalResults['simpleGCR']['rMA'][normName] = dict()
            finalResults['simpleGCR']['rMS'][normName] = dict()
        for gtmAbst in GCRPlus.gtmReductionStrings():
            finalResults['simpleGCR'][gtmAbst] = dict()
            for normName in GCRPlus.getAllNeededReductionNames():
                finalResults['simpleGCR'][gtmAbst][normName] = dict()

        for sKeys in finalResults['simpleGCR'].keys():
            for normName in GCRPlus.getAllNeededReductionNames():
                finalResults['simpleGCR'][sKeys][normName]['gcr'] = []
                finalResults['simpleGCR'][sKeys][normName]['acc'] =[]
                finalResults['simpleGCR'][sKeys][normName]['predicsion'] =[]
                finalResults['simpleGCR'][sKeys][normName]['recall'] =[]
                finalResults['simpleGCR'][sKeys][normName]['f1'] =[]
                finalResults['simpleGCR'][sKeys][normName]['predictResults'] =[]
                finalResults['simpleGCR'][sKeys][normName]['giniScore'] =[]
                finalResults['simpleGCR'][sKeys][normName]['aucScoreGlobal'] = []
                finalResults['simpleGCR'][sKeys][normName]['aucScoreLocal'] = []
                finalResults['simpleGCR'][sKeys][normName]['confidence'] = []

        #[acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc *3

        finalResults['gcr'] = dict()
        for s in res['saliency'].keys():
            finalResults[s] = dict()

            finalResults[s]['saliency'] = []
            finalResults[s]['gtm rAvg acc'] = []
            finalResults[s]['gtm sum acc'] = [] 
         

            finalResults['gcr'][s] = dict()
            for mode in modes:
                finalResults['gcr'][s][mode] = dict()
                finalResults['gcr'][s][mode]['rMA'] = dict()
                finalResults['gcr'][s][mode]['rMS'] = dict()
                for normName in GCRPlus.getAllNeededReductionNames():
                    finalResults['gcr'][s][mode]['rMA'][normName] = dict()
                    finalResults['gcr'][s][mode]['rMS'][normName] = dict()

                for gtmAbst in GCRPlus.gtmReductionStrings():
                    finalResults['gcr'][s][mode][gtmAbst] = dict()
                    for normName in GCRPlus.getAllNeededReductionNames():
                        finalResults['gcr'][s][mode][gtmAbst][normName] = dict()

            for sKeys in finalResults['gcr'][s][0].keys():
                for normName in GCRPlus.getAllNeededReductionNames():
                    for mode in modes:
                        finalResults['gcr'][s][mode][sKeys][normName]['gcr'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['acc'] =[]
                        finalResults['gcr'][s][mode][sKeys][normName]['predicsion'] =[]
                        finalResults['gcr'][s][mode][sKeys][normName]['recall'] =[]
                        finalResults['gcr'][s][mode][sKeys][normName]['f1'] =[]
                        finalResults['gcr'][s][mode][sKeys][normName]['confidence'] =[]
                        finalResults['gcr'][s][mode][sKeys][normName]['predictResults'] =[]
                        finalResults['gcr'][s][mode][sKeys][normName]['giniScore'] =[]
                        finalResults['gcr'][s][mode][sKeys][normName]['aucScoreGlobal'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['aucScoreLocal'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['trainGlobal'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['trainLocal'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['trainGlobalR'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['trainLocalR'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['trainGlobalGini'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['trainLocalGini'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['globalGCRAUC'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['globalGCRAUCReduction'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['globalGCRAUCGini'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['localGCRAUC'] = []

                        finalResults['gcr'][s][mode][sKeys][normName]['globalGCRAUCGini'] = []
                        finalResults['gcr'][s][mode][sKeys][normName]['globalGCRAUCConfidence'] = []           


        classifiers = [
            RandomForestClassifier(),
            AdaBoostClassifier(),
            #LinearDiscriminantAnalysis(),
            #QuadraticDiscriminantAnalysis(),
            RotationForest()
        ]
        classifierNames= [
            "RandomForestClassifier",
            "AdaBoostClassifier",
            #"LDA",
            #"QDA",
            "RotationForest"
        ]
        for cn in classifierNames: 
            printCheck(self.printName, cn + 'start')
            finalResults[cn + ' acc'] = [] 
            finalResults[cn + ' prec'] = []
            finalResults[cn + ' rec'] = []
            finalResults[cn + ' f1'] = []
            finalResults[cn + ' fidelity acc'] = []
            finalResults[cn + ' fidelity prec'] = []
            finalResults[cn + ' fidelity rec'] = []
            finalResults[cn + ' fidelity f1'] = []

        for ti, x_train1 in enumerate(res['results']['trainData']):

            newT = np.squeeze(x_train1) 
            newG = np.squeeze(res['results']['testData'][ti])
            trainTargety = np.argmax(res['results']['trainTarget'][ti], axis=1)
            testTargety = np.argmax(res['testTarget'], axis=1)
            trainPredy = np.argmax(res['results']['trainPred'][ti], axis=1)
            testPredy = np.argmax(res['results']['testPred'][ti], axis=1)

            for ci, clf in enumerate(classifiers):

                clf.fit(newT, trainTargety)
                treePredict = clf.predict(newG)

                #finalResults['treeImportances'].append(clf.feature_importances_)
                finalResults[classifierNames[ci] + ' acc'].append(metrics.accuracy_score(treePredict, testTargety))
                finalResults[classifierNames[ci] + ' prec'].append( metrics.precision_score(treePredict, testTargety, average='macro'))
                finalResults[classifierNames[ci] + ' rec'].append( metrics.recall_score(treePredict, testTargety, average='macro'))
                finalResults[classifierNames[ci] + ' f1'].append(metrics.f1_score(treePredict, testTargety, average='macro'))

                #if len(set(testPredy)) > 1:
                clf.fit(newT, trainPredy)
                treePredict = clf.predict(newG)

                finalResults[classifierNames[ci] + ' fidelity acc'].append(metrics.accuracy_score(treePredict, testPredy))
                finalResults[classifierNames[ci] + ' fidelity prec'].append( metrics.precision_score(treePredict, testPredy, average='macro'))
                finalResults[classifierNames[ci] + ' fidelity rec'].append( metrics.recall_score(treePredict, testPredy, average='macro'))
                finalResults[classifierNames[ci] + ' fidelity f1'].append(metrics.f1_score(treePredict, testPredy, average='macro'))


                printCheck(self.printName, classifierNames[ci] +classifierNames[ci] + ' acc:'  + str(metrics.accuracy_score(treePredict, testTargety)))

                printCheck(self.printName, classifierNames[ci] +classifierNames[ci] + ' fidelity acc:'  + str(metrics.accuracy_score(treePredict, testPredy)))
                #else:
                #    print('Model predicted only one class!')
                #    finalResults[classifierNames[ci] + ' fidelity acc'].append([])
                #     finalResults[classifierNames[ci] + ' fidelity prec'].append([])
                #     finalResults[classifierNames[ci] + ' fidelity rec'].append([])
                #    finalResults[classifierNames[ci] + ' fidelity f1'].append([])



        for s in res['results']['testPred']:
            finalResults['model acc'].append(metrics.accuracy_score(np.argmax(s, axis=1), np.argmax(tl, axis=1)))
            finalResults['model prec'].append(metrics.precision_score(np.argmax(s, axis=1), np.argmax(tl, axis=1), average='macro'))
            finalResults['model rec'].append(metrics.recall_score(np.argmax(s, axis=1), np.argmax(tl, axis=1), average='macro'))
            finalResults['model f1'].append(metrics.f1_score(np.argmax(s, axis=1), np.argmax(tl, axis=1), average='macro'))

        rMGKeys = dict()
        indexP = 0
        valuesA = self.valuesA
        valuesA.append(-2)
        print(valuesA)
        for k in valuesA:
            rMGKeys[round(float(k),4)] = indexP
            indexP += 1

        print('starting GCR')
        self.printTime()
        doBaseGCR = True
        if doBaseGCR:
            for ti, trainData in enumerate(fullResults['results']['trainData']):

                x_test = fullResults['results']['testData'][ti]



                ix_train = []
                ix_test = []

                printCheck(self.printName, 'mapping train test data start')

                for xt in x_test:
                    ix = []
                    for ixt in xt.squeeze():
                        ix.append(rMGKeys[round(float(ixt),4)])
                    ix_test.append(ix)
                ix_test = np.array(ix_test)

                for xt in trainData:
                    ix = []
                    for ixt in xt.squeeze():
                        ix.append(rMGKeys[round(float(ixt),4)])
                    ix_train.append(ix)
                ix_train = np.array(ix_train)

                printCheck(self.printName, 'train combis start')
                trainPredictions = np.argmax(fullResults['results']['trainPred'][ti],axis=1)
                trainCombis = [np.array(list(itertools.product([trainPredictions[i]],s,s))) for i, s in enumerate(ix_train)]
                
                trainCombis = np.concatenate(np.array(trainCombis))

                printCheck(self.printName, 'test combis start')
                testCombis = [list(itertools.product(s,s)) for i, s in enumerate(ix_test)]
                testCombis = np.concatenate(np.array(testCombis))

                printCheck(self.printName, 'ranges big start')
                dataLen = len(ix_test[0])
                ranges= np.array(list(itertools.product(range(len(trainPredictions)),range(dataLen),range(dataLen))))
                printCheck(self.printName, 'ranges small start')
                rangesSmall = np.array(list(itertools.product(range(0,dataLen),range(0,dataLen))))


                for normName in GCRPlus.getAllNeededReductionNames():
                


                    #x_train1 = x_train1.squeeze()
                    
                    #calll other auc
                    printCheck(self.printName, 'gcr simple start')
                    simpleGCRResults = sh.doSimpleGCR(ix_train, fullResults['results']['trainPred'][ti], ix_test, fullResults['results']['testPred'][ti], num_of_classes, nrSymbols, trainCombis, ranges, rangesSmall, testCombis, range(dataLen), reductionName=normName, doMetrics=False)

                    #finalResults['simpleGCR']['rMS'][normName]['gcr'].append(simpleGCRResults[-1])
                    finalResults['simpleGCR']['rMS'][normName]['acc'].append(simpleGCRResults[0][0][0])
                    finalResults['simpleGCR']['rMS'][normName]['predicsion'].append(simpleGCRResults[0][0][1])
                    finalResults['simpleGCR']['rMS'][normName]['recall'].append(simpleGCRResults[0][0][2])
                    finalResults['simpleGCR']['rMS'][normName]['f1'].append(simpleGCRResults[0][0][3])
                    #finalResults['simpleGCR']['rMS']['predictResults'].append(simpleGCRResults[0][3])
                    finalResults['simpleGCR']['rMS'][normName]['giniScore'].append(simpleGCRResults[0][5])
                    finalResults['simpleGCR']['rMS'][normName]['confidence'].append(simpleGCRResults[0][4])
                    #finalResults['simpleGCR']['rMS'][normName]['aucScoreGlobal'].append(simpleGCRResults[0][6]) 
                    #finalResults['simpleGCR']['rMS'][normName]['aucScoreLocal'].append(simpleGCRResults[0][7])

                    #finalResults['simpleGCR']['rMA'][normName]['gcr'].append(simpleGCRResults[-2])
                    finalResults['simpleGCR']['rMA'][normName]['acc'].append(simpleGCRResults[1][0][0])
                    finalResults['simpleGCR']['rMA'][normName]['predicsion'].append(simpleGCRResults[1][0][1])
                    finalResults['simpleGCR']['rMA'][normName]['recall'].append( simpleGCRResults[1][0][2])
                    finalResults['simpleGCR']['rMA'][normName]['f1'].append(simpleGCRResults[1][0][3])
                    #finalResults['simpleGCR']['rMA']['predictResults'].append(simpleGCRResults[1][3])
                    finalResults['simpleGCR']['rMA'][normName]['giniScore'].append(simpleGCRResults[1][5])
                    finalResults['simpleGCR']['rMA'][normName]['confidence'].append(simpleGCRResults[1][4])
                    #finalResults['simpleGCR']['rMA'][normName]['aucScoreGlobal'].append(simpleGCRResults[1][6])
                    #finalResults['simpleGCR']['rMA'][normName]['aucScoreLocal'].append(simpleGCRResults[1][7])

                    for gtmAbstact in GCRPlus.gtmReductionStrings():
                        #finalResults['simpleGCR'][gtmAbstact][normName]['gcr'].append(simpleGCRResults[-4][gtmAbstact])
                        finalResults['simpleGCR'][gtmAbstact][normName]['acc'].append(simpleGCRResults[2][gtmAbstact]['performance'][0])
                        finalResults['simpleGCR'][gtmAbstact][normName]['predicsion'].append(simpleGCRResults[2][gtmAbstact]['performance'][1])
                        finalResults['simpleGCR'][gtmAbstact][normName]['recall'].append(simpleGCRResults[2][gtmAbstact]['performance'][2])
                        finalResults['simpleGCR'][gtmAbstact][normName]['f1'].append(simpleGCRResults[2][gtmAbstact]['performance'][3])
                        #finalResults['simpleGCR'][gtmAbstact]['predictResults'].append(simpleGCRResults[2][gtmAbstact]['performance'][3])
                        finalResults['simpleGCR'][gtmAbstact][normName]['giniScore'].append(simpleGCRResults[2][gtmAbstact]['giniScore'])
                        finalResults['simpleGCR'][gtmAbstact][normName]['confidence'].append(simpleGCRResults[2][gtmAbstact]['confidence'])
                        #finalResults['simpleGCR'][gtmAbstact][normName]['aucScoreGlobal'].append(simpleGCRResults[2][gtmAbstact]['aucScoreGlobal'])
                        #finalResults['simpleGCR'][gtmAbstact][normName]['aucScoreLocal'].append(simpleGCRResults[2][gtmAbstact]['aucScoreLocal'])
                    

                    for mode in modes: # TODO modes nur die richtigen pro methode nehmen! Irgendwo eine Map machen!
                        for k in res['saliency'].keys():
                            #sd = res['saliency'][k]['outTest'][ti]
                            tsd = res['saliency'][k]['outTrain'][ti]
                            tesd = res['saliency'][k]['outTest'][ti]
                            print(ix_train.shape)
                            print(tsd.shape)

                            do3DData = False
                            do2DData = False
                            if len(tsd.shape) > 3:
                                do3DData = True
                            elif len(tsd.shape) > 2:
                                do2DData = True


                            salTr = res['saliency'][k]['outTrain'][ti]
                            print(salTr.shape)
                            salTr = sh.reduceMap(salTr, do3DData=do3DData, do3D2Step=False, do3rdStep=do2DData, axis1= 2, axis2=0, axis3=1, op1='sum',op2='sum',op3='sum').squeeze()
                            print(salTr.shape)
                            saliencyCombis = [np.array(list(itertools.product([trainPredictions[i]],s,s))) for i, s in enumerate(salTr)]
                            saliencyCombis = np.concatenate(np.array(saliencyCombis))


                            printCheck(self.printName, 'gcr full start')
                            sh.doLasaAuc3DGCR(finalResults['gcr'][k][mode], tsd, tesd, ix_train, res['results']['trainPred'][ti], ix_test, res['results']['testPred'][ti], num_of_classes, nrSymbols, trainCombis, ranges, rangesSmall, testCombis, range(dataLen), saliencyCombis, reductionName=normName, mode=modesMap[k], do3DData=do3DData, do2DData=do2DData)


                            #TODO ganzen prozess nochmal überprüfen! Auch mit dem preprocessing! Wird alles richtig weitergegeben!
                            #TODO genau überlegen was für Plots ich machen will!:
                            #- SimpleGCR Performance vs Fidelity vs best local removal Fidelity vs best global removal Fidelity vs best local+global Fidelity? #NOTE: tGCR hier interessant weil nun normalisiert!!!
                            #-- NOTE atm könnte ich zu bestoverall nicht angeben: Gini + Certainty NOTE-> best global + overall
                            #- SimpleGCR Performance vs Fidelity vs Alternative Models Fidelity!!! (auf test nur?)
                            #- Local AUC + Matching SimpleGCR AUC and (best?) global AUC + matchingSimpleGCR? #NOTE andere modelle nicht AUC berechnen da hier "removal" idr Fragwürdig ist!
                            #-- Analysis if specific reduction might remove noise and thus help specific methods? ie IG might work better on higher scores?
                            #- Certainty AUC (Nur best oder nur first?) #TODO add SimpleGCR Certainty!!! und ggf auch bei tGCR?
                            #- Performance der Modele (solo und in kombination aller parameter?) vs Performance andere basic methoden?  #TODO FIdelity und Performance AlternativeModels einbauen!
                            #- beating the different baselines! + avg diff
                            #- NOTE somehow reconstruction of the data?
                            #- Robustness und andere Quantus metricen ergibt nur bedingt Sinn weil Discrete features + interpretable methode und kein NN! (NOTE vll trotzdem das mit leichten data permutations antesten local bei mir? Um zu schauen ob die Saliencymaps stabil bleiben?)
                            # - NOTE zwischen good und bad performing models unterscheiden?
                            # - NOTE Future work: Additional Analysis how the models behaviour change on permutation!
                            # - NOTE Distance analysis kann ich dafür nutzen zu sagen, die SAX approaches sind oft ähnlich, obwohl data changes? 


        printCheck(self.printName, 'finished all')
        print('finished basic GCR')
        self.printTime()
        #for t in thresholds:
        #    self.thresholdsProcess(res, t, finalResults, doGCR=True)
                
        saveName = pt.getWeightName(self.dsName, self.dataName, batch_size, epochs, numOfLayers, header, dmodel, dfff, self.symbolsCount, self.symbolicStrategy ,self.symbolificationStrategy, self.numberFeatures, self.doSymbolify, self.doFeatureExtraction, learning = False, results = True, resultsPath=filteredResults)

        helper.save_obj(finalResults, str(saveName))

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
