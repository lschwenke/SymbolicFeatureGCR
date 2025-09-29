import numpy as np
import dill as pickle
import math
import pandas as pd
import os
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from pyts.approximation import SymbolicAggregateApproximation
from pyts.approximation import MultipleCoefficientBinning
import itertools

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfel import time_series_features_extractor
from tsfel import get_features_by_domain
import pycatch22

def doCombiStep(step, field, axis) -> np.ndarray:
    if(step == 'max'):
        return np.max(field, axis=axis)
    elif (step == 'sum'):
        return np.sum(field, axis=axis)

#flatten an 3D np array
def flatten(X, pos = -1):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    if pos == -1:
        pos = X.shape[1]-1
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, pos, :]
    return(flattened_X)

# Scale 3D array. X = 3D array, scalar = scale object from sklearn. Output = scaled 3D array.
def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

# Symbolize a 3D array. X = 3D array, scalar = SAX symbolizer object. Output = symbolic 3D string array.
def symbolize(X, scaler):
    X_s = scaler.transform(X)

    return X_s

# translate the a string [a,e] between 
def trans(val, vocab) -> float:
    for i in range(len(vocab)):
        if val == vocab[i]:
            halfSize = (len(vocab)-1)/2
            return (i - halfSize) / halfSize
    return -2

def transArray(val, vocab) -> float:
    for i in range(len(vocab)):
        if np.array_equal(val,vocab[i]):
            halfSize = (len(vocab)-1)/2
            return (i - halfSize) / halfSize
    return -2

def getMapValues(size):
    vMap = []
    for i in range(size):
        halfSize = (size-1)/2
        vMap.append(round((i - halfSize) / halfSize, 4))
    return vMap

def symbolizeOrdinalPatternTrans(X, vocab, bins = 5):
    X_s = X
    X_o = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):

        
        for j in range(X.shape[1]):
            X_o[i][j] = transArray(X_s[i][j], vocab)
    return X_o

def symbolizeTransVocab(X, scaler, vocab):
    nanValues = np.argwhere(np.isnan(X))
    X[nanValues] = 0
    X_s = scaler.transform(X)
    for i in range(X.shape[0]):
        X = X.astype(float)
        
        
        for j in range(X.shape[1]):
            X[i][j] = trans(X_s[i][j], vocab)
    return X

def symbolizeTrans(X, scaler, bins = 5):
    vocab = scaler._check_params(bins)
    X_s = scaler.transform(X)
    for i in range(X.shape[0]):
        X = X.astype(float)
        
        
        for j in range(X.shape[1]):
            X[i][j] = trans(X_s[i][j], vocab)
    return X

def symbolizeTrans2(X, scaler, bins = 5):
    vocab = scaler._check_params(bins)
    for i in range(X.shape[0]):
        X_s = X.astype(str) 
        z1 = scaler.transform(np.array([X[i, :, :][:,0]]))
        X_s[i, :, :][:,0] = z1
        for j in range(X.shape[1]):
            X[i][j][0] = trans(X_s[i][j][0], vocab)
    return X

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        np.save(f, obj)

def save_obj2(obj, name, bigLoad=False):
    if bigLoad:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(f, obj, protocol=4)
    else:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(f, obj)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return np.load(f, allow_pickle=True)
		
def load_obj2(name, bigLoad=False):
    if bigLoad:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f, protocol=4)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    
def truncate(n):
    return int(n * 1000) / 1000

def ceFull(data):
    complexDists = []
    for d in data:
        complexDists.append(ce(d))
        
    return complexDists, np.mean(complexDists) 

def ce(data):
    summer = 0
    for i in range(len(data)-1):
        summer += math.pow(data[i] - data[i+1], 2)
    return math.sqrt(summer)


def modelFidelity(modelPrediction, interpretationPrediction):
    summer = 0
    for i in range(len(modelPrediction)):
        if modelPrediction[i] == interpretationPrediction[i]:
            summer += 1
    return summer / len(modelPrediction)

def collectLAAMs(earlyPredictor, x_test, order, step1, step2):
    limit = 500
    attentionQ0 = []
    attentionQ1 = []
    attentionQ2 = []

    for bor in range(int(math.ceil(len(x_test)/limit))):
        attOut = earlyPredictor.predict([x_test[bor*limit:(bor+1)*limit]])
        attentionQ0.extend(attOut[0]) 
        attentionQ1.extend(attOut[1])

        if len(attentionQ2) == 0:
            attentionQ2 = attOut[2]
        else:
            for k in range(len(attentionQ2)):
                
                attentionQ2[k] = np.append(attentionQ2[k], attOut[2][k], 0)
    
    attentionFQ = [np.array(attentionQ0), np.array(attentionQ1), np.array(attentionQ2)]
    
    if(order == 'lh'):
        axis1 = 0
        axis2 = 1
    elif(order == 'hl'):
        axis1 = 2
        axis2 = 0
    else:
        axis1 = 0
        axis2 = 1   

    attentionFQ[1] = doCombiStep(step1, attentionFQ[2], axis1)
    attentionFQ[1] = doCombiStep(step2, attentionFQ[1], axis2) 

    return attentionFQ[1]


def laamConsistency(laams, fromIndex, consistancyLabels):
    results = dict()
    innerFolddistance = dict()
    outerdistance = dict()
    innerClassDistance = dict()
    for combi in laams.keys():
        innerFolddistance[combi] = []
        outerdistance[combi] = []
        innerClassDistance[combi] = []
        for j in range(len(laams[combi][fromIndex])):
            outerdistance[combi].append([])
            for fold in range(len(laams[combi])):
                if fold != fromIndex:
                    outerdistance[combi][j].append(matrixEucDistance(laams[combi][fromIndex][j],laams[combi][fold][j]))

        for fold in range(len(laams[combi])): 
            innerFolddistance[combi].append([])
            for j in range(len(laams[combi][fold])):
                if j != fromIndex and consistancyLabels[j] != consistancyLabels[fromIndex]:
                    innerFolddistance[combi][fold].append(matrixEucDistance(laams[combi][fold][fromIndex],laams[combi][fold][j]))
        
        for fold in range(len(laams[combi])): 
            innerClassDistance[combi].append([])
            for j in range(len(laams[combi][fold])):
                if j != fromIndex and consistancyLabels[j] == consistancyLabels[fromIndex]:
                    innerClassDistance[combi][fold].append(matrixEucDistance(laams[combi][fold][fromIndex],laams[combi][fold][j]))
    
    #from one sample to another of a different class
    results["innerFold"] = innerFolddistance
    #from one sample to another of the same class
    results["innerClass"] = innerClassDistance
    #same sample between folds
    results["outer"] = outerdistance
    return results

def matrixEucDistance(matrix1, matrix2):
    summer = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            summer += math.pow(matrix1[i][j] - matrix2[i][j], 2)
    return math.sqrt(summer)

def confidenceGCR(bestScores, correctness):
    top80Len = int(len(bestScores) * 0.80)
    top50Len = int(len(bestScores) * 0.50)
    top20Len = int(len(bestScores) * 0.20)
    top10Len = int(len(bestScores) * 0.10)
    
    top80Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top80Len:]
    top50Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top50Len:]
    top20Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top20Len:]
    top10Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top10Len:]

    nCorrectness = np.array(correctness)
    top80Acc = (sum(nCorrectness[top80Ind])/len(nCorrectness[top80Ind]))
    top50Acc = (sum(nCorrectness[top50Ind])/len(nCorrectness[top50Ind]))
    top20Acc = (sum(nCorrectness[top20Ind])/len(nCorrectness[top20Ind]))
    top10Acc = (sum(nCorrectness[top10Ind])/len(nCorrectness[top10Ind]))
    
    return top80Acc, top50Acc, top20Acc, top10Acc

def confidenceGCR2(bestScores, correctness, steps):
    results = []
    nCorrectness = np.array(correctness)
    step = 1
    while step > 0:
        topLen = int(len(bestScores) * step)
        topInd = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-topLen:]
        
        results.append(sum(nCorrectness[topInd])/len(nCorrectness[topInd]))
        step = step - (1/steps)
    return results

def fidelityConfidenceGCR(bestScores, correctness, saxResults):
    top80Len = int(len(bestScores) * 0.80)
    top50Len = int(len(bestScores) * 0.50)
    top20Len = int(len(bestScores) * 0.20)
    top10Len = int(len(bestScores) * 0.10)
    
    top80Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top80Len:]
    top50Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top50Len:]
    top20Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top20Len:]
    top10Ind = sorted(range(len(bestScores)), key=lambda x: bestScores[x])[-top10Len:]

    nCorrectness = np.array(correctness)
    nSaxResults = np.array(saxResults)
    top80Fidelity = modelFidelity(nCorrectness[top80Ind], nSaxResults[top80Ind])
    top50Fidelity = modelFidelity(nCorrectness[top50Ind], nSaxResults[top50Ind])
    top20Fidelity = modelFidelity(nCorrectness[top20Ind], nSaxResults[top20Ind])
    top10Fidelity = modelFidelity(nCorrectness[top10Ind], nSaxResults[top10Ind])


    return top80Fidelity, top50Fidelity, top20Fidelity, top10Fidelity


def y_to_tsf(y):
    """ Transform label data into TSFresh format """
    return np.asarray([np.argmax(row) for row in y])


def to_df_tsfresh(X):
    """ Transform data into DataFrame suitable for TSFresh """
    X = X.squeeze()
   
    num_time_series = X.shape[0]
    num_timestamps = X.shape[1]

    ids = []
    times = []
    values = []

    id = 1
    for time_series in range(num_time_series):
        time = 1
        for timestamp in range(num_timestamps):
            ids.append(id)
            times.append(time)
            values.append(X[time_series][timestamp])
            time += 1
        id += 1

    data = {
        'id': ids,
        'time': times,
        'value': values
    }

    df = pd.DataFrame(data)

    return df

def tsf_remove_constant_features(features):
    """ Remove constant features from TSFresh features """
    feat = features
    for key in features.keys():
        if features[key].max() - features[key].min() == 0:
            del feat[key]
    return feat

def generateNoise(x_train1, y_train, y_trainy, numberSymbols, copy=100):
    outTrain = x_train1.copy()
    outLables = y_train.copy()
    outyLables = y_trainy.copy()
    variation = 2 / ((numberSymbols-1) * 2)
    for i in range(copy):
        randomNoise = (np.random.uniform(-1 * variation, variation, size=x_train1.shape))
        newTrain = x_train1.copy() + randomNoise
        outTrain = np.concatenate((outTrain, newTrain))
        outLables = np.concatenate((outLables, y_train.copy()))
        outyLables = np.concatenate((outyLables, y_trainy.copy()))
    return outTrain, outLables, outyLables


def transformOrdinalpatterns(x_train1, pLen = 3,steps = 1, minusValues=False, doSymbolize=True):


    px_train = np.zeros((len(x_train1), len(range(0,len(x_train1[0])- (pLen * steps), steps)), pLen))
    if minusValues:
        px_train = px_train - (1 * (pLen-1) /2)
    pos = 0
    for pi in range(0,len(x_train1[0])- (pLen * steps), steps):
        
        n2 = x_train1[:,pi+steps]
        n3 = x_train1[:,pi+(steps*2)]
        for si in range(pLen):
            n1 = x_train1[:,pi + (steps * si)]
            for si2 in range(pLen):
                n2 = x_train1[:,pi + (steps * si2)]
                px_train[:,pos,si] += n1 > n2
        
        pos += 1

    
    if pLen == 3:
        vocab = list(itertools.product([0,1,2], [0,1,2],[0,1,2]))
    if pLen == 5:
        vocab = list(itertools.product([0,1,2,3,4], [0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]))
    if pLen == 7:
        vocab = list(itertools.product([0,1,2,3,4,5,6], [0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6], [0,1,2,3,4,5,6], [0,1,2,3,4,5,6]))
    if pLen == 9:
        vocab = list(itertools.product([0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5,6,7,8]))

    if doSymbolize:
        px_train = symbolizeOrdinalPatternTrans(px_train, vocab, bins = pLen)    

    return px_train

def drop_duplicate_features_tsfel(df):
    """ Drop duplicate features in tsfel """
    df = df.drop(columns=['0_Max', '0_Spectral skewness', '0_Root mean square', '0_Mean absolute diff', '0_Neighbourhood peaks', '0_Slope', '0_Mean',
                          '0_Sum absolute diff', '0_Absolute energy', '0_Mean diff', '0_Median', '0_Standard deviation', '0_Min', '0_Zero crossing rate',
                          '0_Variance', '0_Positive turning points'])
    return df

def combine_features(features_a, features_b):
    """ Combine two feature matrices """
    combined_features = pd.concat([features_a, features_b], axis=1)
    return combined_features

def y_to_tsf_top(y, features):
    """ Transform label data into TSFresh top format """
    y_series = pd.Series(y, index=features.index)
    return y_series

def stumpy_to_tsfresh_df(data):
    """ Convert stumpy data to tsfresh DataFrame format """
    df = pd.DataFrame(data)

    return df

def mcb_to_tsfresh_df(data, columns):
    """ Convert MCB data to tsfresh DataFrame format """
    df = pd.DataFrame(data, columns=columns)


    return df


def preprocessDataWithFeatures(x_train1, x_val, X_test, y_train1, y_val, y_test, y_trainy, y_testy, binNr, symbolsCount, dataName, strategy='quantile', symbolicMethod = 'mcb', symbolicMethod2='sax', symbolicStrategy2='uniform', useEmbed = False, useSaves = False, doSymbolify = True, multiVariant=False, doScaling=True, doFeatureExtraction=True, n=200, doSelect=True, doDataAugmentation=False, augmentationNumer=100, doOrdinalPatterns=False, doConcatinate = False):    
    
    x_test = X_test.copy()
    
    if(useEmbed):
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount) + '+embedding'
    else:
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount)
    fileExists = os.path.isfile(processedDataName +'.pkl')

    if(fileExists and useSaves):
        print('found file! Start loading file!')
        res = load_obj(processedDataName)


        for index, v in np.ndenumerate(res):
            print(index)
            res = v
        res.keys()

        x_train1 = res['X_train']
        x_test = res['X_test']
        x_val = res['X_val']
        X_train_ori = res['X_train_ori']
        X_test_ori = res['X_test_ori']
        y_trainy = res['y_trainy']
        y_train1 = res['y_train']
        y_test = res['y_test']
        y_testy = res['y_testy']
        y_val = res['y_val']
        X_val_ori = res['X_val_ori']
        print(x_test.shape)
        print(x_train1.shape)
        print(y_test.shape)
        print(y_train1.shape)
        print('SHAPES loaded')
        
    else:
        print('SHAPES:')
        print(x_test.shape)
        print(x_train1.shape)
        print(x_val.shape)
        print(y_test.shape)
        print(y_train1.shape)

        x_train1 = x_train1.squeeze()
        x_val = x_val.squeeze()
        x_test = x_test.squeeze()
        
        trainShape = x_train1.shape
        valShape = x_val.shape
        testShape = x_test.shape
        
        if multiVariant:
            print('NOT IMPLEMENTED')
            

        else:    
            if doScaling:
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train1 = scaler.transform(x_train1.reshape(-1, 1)).reshape(trainShape)
                x_val = scaler.transform(x_val.reshape(-1, 1)).reshape(valShape)
                x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(testShape)

            X_test_ori = x_test.copy()
            X_val_ori = x_val.copy()
            X_train_ori = x_train1.copy()



            if doFeatureExtraction:

                if True:

                    doOnlyFresh= False
                    
                    if doOnlyFresh:
                        x_train1_df = to_df_tsfresh(x_train1)
                        extracted_features_train1 = extract_features(x_train1_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
                        selected_features_train1 = select_features(extracted_features_train1, y_to_tsf(y_train1), ml_task='classification', multiclass=False)
                        if selected_features_train1.empty: # No relevant features selected
                            return False
                        x_train1 = selected_features_train1.to_numpy()

                        x_val_df = to_df_tsfresh(x_val)
                        extracted_features_val = extract_features(x_val_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
                        selected_features_val = extracted_features_val[selected_features_train1.columns]  # Only extract features from the val and val set that are relevant in the train set
                        x_val = selected_features_val.to_numpy()

                        x_test_df = to_df_tsfresh(x_test)
                        extracted_features_test = extract_features(x_test_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute)
                        selected_features_test = extracted_features_test[selected_features_train1.columns]  # Only extract features from the val and val set that are relevant in the train set
                        x_test = selected_features_test.to_numpy()
                        
                    else: 


                        
                        x_train1_df = to_df_tsfresh(x_train1)
                        x_val_df = to_df_tsfresh(x_val)
                        x_test_df = to_df_tsfresh(x_test)
                        extracted_features_train1_tsfresh = extract_features(x_train1_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute).reset_index().drop(columns=['index'])
                        extracted_features_val_tsfresh = extract_features(x_val_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute).reset_index().drop(columns=['index'])
                        extracted_features_test_tsfresh = extract_features(x_test_df, column_id='id', column_sort='time', column_value='value', default_fc_parameters=ComprehensiveFCParameters(), impute_function=impute).reset_index().drop(columns=['index'])
                    
                        # extract tsfel features
                        x_train1_tsfel = np.expand_dims(x_train1, axis=2)
                        x_val_tsfel = np.expand_dims(x_val, axis=2)
                        x_test_tsfel = np.expand_dims(x_test, axis=2)
                        cfg = get_features_by_domain() # Extracts the temporal, statistical and spectral feature sets.
                        extracted_features_train1_tsfel = time_series_features_extractor(cfg, x_train1_tsfel)
                        extracted_features_val_tsfel = time_series_features_extractor(cfg, x_val_tsfel)
                        extracted_features_test_tsfel = time_series_features_extractor(cfg, x_test_tsfel)
                    
                        # drop the features that occur in tsfresh and tsfel
                        extracted_features_train1_tsfel = drop_duplicate_features_tsfel(extracted_features_train1_tsfel)
                        extracted_features_val_tsfel = drop_duplicate_features_tsfel(extracted_features_val_tsfel)
                        extracted_features_test_tsfel = drop_duplicate_features_tsfel(extracted_features_test_tsfel)


            
                        # combine the features
                        combined_features_train1 = combine_features(extracted_features_train1_tsfresh, extracted_features_train1_tsfel)
                        combined_features_val = combine_features(extracted_features_val_tsfresh, extracted_features_val_tsfel)
                        combined_features_test = combine_features(extracted_features_test_tsfresh, extracted_features_test_tsfel)


            
                        # filter out constant features
                        combined_features_train1 = tsf_remove_constant_features(combined_features_train1)
                        combined_features_val = combined_features_val[combined_features_train1.columns]
                        combined_features_test = combined_features_test[combined_features_train1.columns]
                    
                        # get catch22 features
                        extracted_features_train1_c22 = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_train1])
                        extracted_features_val_c22 = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_val])
                        extracted_features_test_c22 = np.asarray([pycatch22.catch22_all(ts)['values'] for ts in x_test])
                    
                        # transform catch22 features into dataframe
                        extracted_features_train1_c22_df = mcb_to_tsfresh_df(extracted_features_train1_c22, ['catch'+str(item) for item in np.array(list(range(0,extracted_features_train1_c22.shape[1])))])
                        extracted_features_val_c22_df = mcb_to_tsfresh_df(extracted_features_val_c22, ['catch'+str(item) for item in np.array(list(range(0,extracted_features_val_c22.shape[1])))])
                        extracted_features_test_c22_df = mcb_to_tsfresh_df(extracted_features_test_c22, ['catch'+str(item) for item in np.array(list(range(0,extracted_features_test_c22.shape[1])))])


            
                        # combine the features
                        combined_features_train1 = combine_features(combined_features_train1, extracted_features_train1_c22_df)
                        combined_features_val = combine_features(combined_features_val, extracted_features_val_c22_df)
                        combined_features_test = combine_features(combined_features_test, extracted_features_test_c22_df)
                    
                        # filter out constant features
                        combined_stumpy_train1 = tsf_remove_constant_features(combined_features_train1)
                        combined_stumpy_val = combined_features_val[combined_features_train1.columns]
                        combined_stumpy_test = combined_features_test[combined_features_train1.columns]

                        oriColumns = ['ts'+str(item) for item in np.array(list(range(0,X_train_ori.shape[1])))]
                        if symbolicMethod== 'sax':
                            trainSymbolicIndexes = dict()
                            trainSymbolicIndexes[symbolsCount] = np.array(range(len(X_train_ori)))
                        else:
                            trainSymbolicIndexes = dict()
                            
                            X_train_ori = mcb_to_tsfresh_df(X_train_ori, oriColumns)

                            currentColumns = X_train_ori.columns
                            removedColumns = []

                        
                            print('----')
                            print(X_train_ori.shape)
                            for bins in range(symbolsCount, 1, -1):
                                lowVariance = True
                                while lowVariance:
                                
                                    bins_edges = np.percentile(
                                        X_train_ori[currentColumns], np.linspace(0, 100, bins + 1)[1:-1], axis=0
                                    ).T

                                    print(bins_edges.shape)

                                    low_var_cols = np.unique(np.where(np.diff(bins_edges, axis=0) == 0)[0])
                                    removedColumns = np.concatenate((removedColumns,currentColumns[low_var_cols])) 
                                    currentColumns = np.delete(currentColumns, low_var_cols)
                                    
                                    trainSymbolicIndexes[bins] = X_train_ori.columns.get_indexer(currentColumns) 
                                    


                                    if len(low_var_cols) == 0:
                                        lowVariance = False    

                                currentColumns = removedColumns
                                removedColumns = []
                                if len(currentColumns) == 0:
                                        break
                            
                            if len(currentColumns) != 0:
                                print('LEFT OVERS!!!!!!')
                                tsi = X_train_ori.columns.get_indexer(currentColumns)
                                x_train1[:,tsi] = 0
                                x_val[:,tsi] = 0
                                x_test[:,tsi] = 0


                            X_train_ori = X_train_ori.to_numpy()
                            
                        firstRun = True
                        for tsik in trainSymbolicIndexes.keys():
                            print(tsik)
                            tsi = trainSymbolicIndexes[tsik]
                            if symbolicMethod == 'mcb':
                                symbolicFitter = MultipleCoefficientBinning(n_bins=tsik, strategy=symbolicStrategy2)
                            elif symbolicMethod == 'sax':
                                symbolicFitter = SymbolicAggregateApproximation(n_bins=tsik, strategy=symbolicStrategy2)
                            else:
                                print('WARNING no such symbolic method! Using default!')
                                symbolicFitter = MultipleCoefficientBinning(n_bins=tsik, strategy=symbolicStrategy2)
                            symbolicFitter.fit(X_train_ori[:,tsi])

                            if firstRun:
                                firstRun = False
                                vocab = symbolicFitter._check_params(tsik)
                            if(useEmbed):
                                X_train_ori[:,tsi] = symbolize(X_train_ori[:,tsi], symbolicFitter)
                                X_val_ori[:,tsi] = symbolize(X_val_ori[:,tsi], symbolicFitter)
                                X_test_ori[:,tsi] = symbolize(X_test_ori[:,tsi], symbolicFitter)
                            else:
                                x_train1[:,tsi] = symbolizeTransVocab(x_train1[:,tsi], symbolicFitter, vocab )
                                x_val[:,tsi] = symbolizeTransVocab(x_val[:,tsi], symbolicFitter, vocab)
                                x_test[:,tsi] = symbolizeTransVocab(x_test[:,tsi], symbolicFitter, vocab)
                        

                        combined_stumpy_train1 = combine_features(combined_stumpy_train1, mcb_to_tsfresh_df(X_train_ori, oriColumns))
                        combined_stumpy_val = combine_features(combined_stumpy_val, mcb_to_tsfresh_df(X_val_ori, oriColumns))
                        combined_stumpy_test = combine_features(combined_stumpy_test, mcb_to_tsfresh_df(X_test_ori, oriColumns))
                        
                        combined_stumpy_train1 = tsf_remove_constant_features(combined_stumpy_train1)
                        combined_stumpy_val = combined_stumpy_val[combined_stumpy_train1.columns]
                        combined_stumpy_test = combined_stumpy_test[combined_stumpy_train1.columns]


                        top_features_train1 = combined_stumpy_train1
                        top_features_val = combined_stumpy_val
                        top_features_test = combined_stumpy_test
                        
                        currentColumns = []
                        if symbolicMethod== 'sax':
                            trainSymbolicIndexes = dict()
                            trainSymbolicIndexes[symbolsCount] = np.array(range(len(top_features_train1.values)))
                        else:
                            trainSymbolicIndexes = dict()
                            currentColumns = top_features_train1.columns
                            removedColumns = []
                            print('----')
                            print(top_features_train1.shape)
                            for bins in range(symbolsCount, 1, -1):
                                lowVariance = True
                                while lowVariance:
                                
                                    bins_edges = np.percentile(
                                        top_features_train1[currentColumns], np.linspace(0, 100, bins + 1)[1:-1], axis=0
                                    ).T

                                    print(bins_edges.shape)

                                    low_var_cols = np.unique(np.where(np.diff(bins_edges, axis=0) == 0)[0])
                                    removedColumns = np.concatenate((removedColumns,currentColumns[low_var_cols])) 
                                    currentColumns = np.delete(currentColumns, low_var_cols)
                                    
                                    trainSymbolicIndexes[bins] = top_features_train1.columns.get_indexer(currentColumns) 
                                    


                                    if len(low_var_cols) == 0:
                                        lowVariance = False    

                                currentColumns = removedColumns
                                removedColumns = []
                                if len(currentColumns) == 0:
                                        break

                        x_train1 = top_features_train1.to_numpy()
                        x_val = top_features_val.to_numpy()
                        x_test = top_features_test.to_numpy()
                        if len(currentColumns) != 0:
                            tsi = top_features_train1.columns.get_indexer(currentColumns) 
                            x_train1[:,tsi] = 0
                            x_val[:,tsi] = 0
                            x_test[:,tsi] = 0
                        



            if doSymbolify:
                firstRun = True

                
                for tsik in trainSymbolicIndexes.keys(): 
                    print(tsik)
                    tsi = trainSymbolicIndexes[tsik]
                    if tsik == 1:
                        x_train1 = np.delete(x_train1, tsi, axis=1)
                    print(tsi)
                    if symbolicMethod == 'mcb':
                        symbolicFitter = MultipleCoefficientBinning(n_bins=tsik, strategy=strategy)
                    elif symbolicMethod == 'sax':
                        symbolicFitter = SymbolicAggregateApproximation(n_bins=tsik, strategy=strategy)
                    else:
                        print('WARNING no such symbolic method! Using default!')
                        symbolicFitter = MultipleCoefficientBinning(n_bins=tsik, strategy=strategy)

                    symbolicFitter.fit(x_train1[:,tsi])

                    if firstRun:
                        firstRun = False
                        vocab = symbolicFitter._check_params(tsik)

                    if(useEmbed):
                        x_train1[:,tsi] = symbolize(x_train1[:,tsi], symbolicFitter)
                        x_val[:,tsi] = symbolize(x_val[:,tsi], symbolicFitter)
                        x_test[:,tsi] = symbolize(x_test[:,tsi], symbolicFitter)
                    else:
                        x_train1[:,tsi] = np.nan_to_num(symbolizeTransVocab(x_train1[:,tsi], symbolicFitter, vocab), nan=-2)
                        x_val[:,tsi] = np.nan_to_num(symbolizeTransVocab(x_val[:,tsi], symbolicFitter, vocab), nan=-2)
                        x_test[:,tsi] = np.nan_to_num(symbolizeTransVocab(x_test[:,tsi], symbolicFitter, vocab), nan=-2)

                if doConcatinate:
                    symbolicMethod = 'sax'
                    if symbolicMethod == 'mcb':
                        symbolicFitter = MultipleCoefficientBinning(n_bins=symbolsCount, strategy='uniform')
                    elif symbolicMethod == 'sax':
                        symbolicFitter = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy='uniform')
                    else:
                        print('WARNING no such symbolic method! Using default!')
                        symbolicFitter = MultipleCoefficientBinning(n_bins=symbolsCount, strategy='uniform')
                    symbolicFitter.fit(X_train_ori)

                    x_train1 = np.concatenate((x_train1, symbolizeTrans(X_train_ori, symbolicFitter, bins = symbolsCount)), axis=1)
                    x_val = np.concatenate((x_val, symbolizeTrans(X_val_ori, symbolicFitter, bins = symbolsCount)), axis=1)
                    x_test = np.concatenate((x_test, symbolizeTrans(X_test_ori, symbolicFitter, bins = symbolsCount)), axis=1)

                    


                if doDataAugmentation:
                    x_train1, y_train1, y_trainy = generateNoise(x_train1, y_train1, y_trainy, symbolsCount, copy=augmentationNumer)
                    x_val, y_val, _ = generateNoise(x_val, y_val, np.array([]), symbolsCount, copy=augmentationNumer)
            else:
                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()

            doSymbolize=True


            
            combined_stumpy_train1 = mcb_to_tsfresh_df(x_train1, columns=combined_stumpy_train1.columns)
            combined_stumpy_val = mcb_to_tsfresh_df(x_val, columns=combined_stumpy_val.columns)
            combined_stumpy_test = mcb_to_tsfresh_df(x_test, columns=combined_stumpy_test.columns)

            relevance_table = calculate_relevance_table(combined_stumpy_train1, y_to_tsf_top(y_trainy, combined_stumpy_train1), ml_task='classification', multiclass=False)
            top_n_features = relevance_table.sort_values("p_value").head(n).index
            top_features_train1 = combined_stumpy_train1[top_n_features]
            top_features_val = combined_stumpy_val[top_n_features]
            top_features_test = combined_stumpy_test[top_n_features]

            top_features_train1 = top_features_train1.dropna(axis='columns')
            top_features_val = top_features_val[top_features_train1.columns]
            top_features_test = top_features_test[top_features_train1.columns]


            x_train1 = top_features_train1.to_numpy()
            x_val = top_features_val.to_numpy()
            x_test = top_features_test.to_numpy()
            

            if doOrdinalPatterns:
                pLen = symbolsCount
                steps = symbolsCount-1
                
                if doConcatinate:
                    """
                    if strategy == 'quantile' and doSymbolify:
                        # Iteratively remove low variance features for quantile binning
                        filter_low_variance = True

                        while filter_low_variance:
                            print('Delete low variance features')
                            print(X_train_ori.shape)
                            bins_edges = np.percentile(
                                X_train_ori, np.linspace(0, 100, symbolsCount + 1)[1:-1], axis=0
                            ).T

                            print(bins_edges.shape)

                            low_var_cols = np.unique(np.where(np.diff(bins_edges, axis=0) == 0)[0])

                            print(low_var_cols.shape)
                            print('Deleting: ', low_var_cols)
                            X_train_ori = np.delete(X_train_ori, low_var_cols, axis=1)
                            X_val_ori = np.delete(X_val_ori, low_var_cols, axis=1)
                            X_test_ori = np.delete(X_test_ori, low_var_cols, axis=1)


                            bins_edges = np.percentile(
                                X_train_ori, np.linspace(0, 100, symbolsCount + 1)[1:-1], axis=0
                            ).T
                            
                            low_var_cols = np.unique(np.where(np.diff(bins_edges, axis=0) == 0)[0])
                            print('Remaining: ', low_var_cols)
                        
                            if len(low_var_cols) == 0:
                                print('Aborting')
                                filter_low_variance = False
                    """

                    #if symbolicMethod == 'mcb':
                    #    symbolicFitter = MultipleCoefficientBinning(n_bins=symbolsCount, strategy=strategy)
                    #elif symbolicMethod == 'sax':
                    symbolicFitter = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy=strategy)
                    #else:
                    #    print('WARNING no such symbolic method! Using default!')
                    #    symbolicFitter = MultipleCoefficientBinning(n_bins=symbolsCount, strategy=strategy)
                    symbolicFitter.fit(X_train_ori)
                    #sax.fit(x_train1)

                    

                    if(useEmbed):
                        x_trainAdd = symbolize(X_train_ori, symbolicFitter)
                        x_valAdd = symbolize(X_val_ori, symbolicFitter)
                        x_testAdd = symbolize(X_test_ori, symbolicFitter)
                    else:
                        x_trainAdd = symbolizeTrans(X_train_ori, symbolicFitter, bins = symbolsCount)
                        x_valAdd = symbolizeTrans(X_val_ori, symbolicFitter, bins = symbolsCount)
                        x_testAdd = symbolizeTrans(X_test_ori, symbolicFitter, bins = symbolsCount)

                    x_trainAdd = transformOrdinalpatterns(x_trainAdd, pLen = pLen,steps =steps, doSymbolize=doSymbolize)
                    x_valAdd = transformOrdinalpatterns(x_valAdd, pLen = pLen,steps = steps, doSymbolize=doSymbolize)
                    x_testAdd = transformOrdinalpatterns(x_testAdd, pLen = pLen,steps = steps, doSymbolize=doSymbolize)   

                    x_train1 = np.concatenate((x_train1, x_trainAdd), axis=1)
                    x_val = np.concatenate((x_val, x_valAdd), axis=1)
                    x_test = np.concatenate((x_test, x_testAdd), axis=1)

                else:
                    x_train1 = transformOrdinalpatterns(x_train1, pLen = pLen,steps =steps, doSymbolize=doSymbolize)
                    x_val = transformOrdinalpatterns(x_val, pLen = pLen,steps = steps, doSymbolize=doSymbolize)
                    x_test = transformOrdinalpatterns(x_test, pLen = pLen,steps = steps, doSymbolize=doSymbolize)      


            x_train1 = np.expand_dims(x_train1, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)   
            X_test_ori = np.expand_dims(X_test_ori, axis=2)   
            X_train_ori = np.expand_dims(X_train_ori, axis=2) 
            X_val_ori = np.expand_dims(X_val_ori, axis=2) 
        
            

        print('saves shapes:')
        print(x_test.shape)
        print(x_train1.shape)

        #save sax results to only calculate them once
        resultsSave = {
            'X_train':x_train1,
            'X_train_ori':X_train_ori,
            'X_test':x_test,
            'X_test_ori':X_test_ori,
            'X_val': x_val,
            'X_val_ori':X_val_ori,
            'y_trainy':y_trainy,
            'y_train':y_train1,
            'y_val': y_val,
            'y_test':y_test,
            'y_testy':y_testy
        }
        save_obj(resultsSave, processedDataName)

    return x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy, top_n_features




def preprocessData(x_train1, x_val, X_test, y_train1, y_val, y_test, y_trainy, y_testy, binNr, symbolsCount, dataName, useEmbed = False, useSaves = False, doSymbolify = True, multiVariant=False):    
    
    x_test = X_test.copy()
    
    if(useEmbed):
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount) + '+embedding'
    else:
        processedDataName = "./saves/"+str(dataName)+ '-bin' + str(binNr) + '-symbols' + str(symbolsCount)
    fileExists = os.path.isfile(processedDataName +'.pkl')

    if(fileExists and useSaves):
        print('found file! Start loading file!')
        res = load_obj(processedDataName)


        for index, v in np.ndenumerate(res):
            print(index)
            res = v
        res.keys()

        x_train1 = res['X_train']
        #x_train1 = res['X_val']
        x_test = res['X_test']
        x_val = res['X_val']
        X_train_ori = res['X_train_ori']
        X_test_ori = res['X_test_ori']
        y_trainy = res['y_trainy']
        y_train1 = res['y_train']
        y_test = res['y_test']
        y_testy = res['y_testy']
        y_val = res['y_val']
        X_val_ori = res['X_val_ori']
        print(x_test.shape)
        print(x_train1.shape)
        print(y_test.shape)
        print(y_train1.shape)
        print('SHAPES loaded')
        
    else:
        print('SHAPES:')
        print(x_test.shape)
        print(x_train1.shape)
        print(x_val.shape)
        print(y_test.shape)
        print(y_train1.shape)

        x_train1 = x_train1.squeeze()
        x_val = x_val.squeeze()
        x_test = x_test.squeeze()
        
        trainShape = x_train1.shape
        valShape = x_val.shape
        testShape = x_test.shape
        
        if multiVariant:
            X_test_ori = x_test.copy()
            X_val_ori = x_val.copy()
            X_train_ori = x_train1.copy()
            for i in range(trainShape[-1]):
                x_train2 = x_train1[:,:,i]
                x_val2 = x_val[:,:,i]
                x_test2 = x_test[:,:,i]
                print('####')
                print(x_train2.shape)

                trainShape2 = x_train2.shape
                valShape2 = x_val2.shape
                testShape2 = x_test2.shape
        
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train2 = scaler.transform(x_train2.reshape(-1, 1)).reshape(trainShape2)##
                x_val2 = scaler.transform(x_val2.reshape(-1, 1)).reshape(valShape2)
                x_test2 = scaler.transform(x_test2.reshape(-1, 1)).reshape(testShape2)

                sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy='uniform')
                sax.fit(x_train2)

                if(useEmbed):
                    x_train2 = symbolize(x_train2, sax)
                    x_val2 = symbolize(x_val2, sax)
                    x_test2 = symbolize(x_test2, sax)
                else:
                    x_train2 = symbolizeTrans(x_train2, sax, bins = symbolsCount)
                    x_val2 = symbolizeTrans(x_val2, sax, bins = symbolsCount)
                    x_test2 = symbolizeTrans(x_test2, sax, bins = symbolsCount)
                print(x_train2.shape)

                x_train1[:,:,i] = x_train2      
                x_val[:,:,i] = x_val2
                x_test[:,:,i] = x_test2

            print(x_train1.shape)
            

        else:    
            if(doSymbolify):
                scaler = StandardScaler()    
                scaler = scaler.fit(x_train1.reshape((-1,1)))
                x_train1 = scaler.transform(x_train1.reshape(-1, 1)).reshape(trainShape)
                x_val = scaler.transform(x_val.reshape(-1, 1)).reshape(valShape)
                x_test = scaler.transform(x_test.reshape(-1, 1)).reshape(testShape)

                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()


                sax = SymbolicAggregateApproximation(n_bins=symbolsCount, strategy='uniform')
                sax.fit(x_train1)

                if(useEmbed):
                    x_train1 = symbolize(x_train1, sax)
                    x_val = symbolize(x_val, sax)
                    x_test = symbolize(x_test, sax)
                else:
                    x_train1 = symbolizeTrans(x_train1, sax, bins = symbolsCount)
                    x_val = symbolizeTrans(x_val, sax, bins = symbolsCount)
                    x_test = symbolizeTrans(x_test, sax, bins = symbolsCount)
            else:
                X_test_ori = x_test.copy()
                X_val_ori = x_val.copy()
                X_train_ori = x_train1.copy()


            x_train1 = np.expand_dims(x_train1, axis=2)
            x_val = np.expand_dims(x_val, axis=2)
            x_test = np.expand_dims(x_test, axis=2)   
            X_test_ori = np.expand_dims(X_test_ori, axis=2)   
            X_train_ori = np.expand_dims(X_train_ori, axis=2) 
            X_val_ori = np.expand_dims(X_val_ori, axis=2) 
            
            

        print('saves shapes:')
        print(x_test.shape)
        print(x_train1.shape)

        #save sax results to only calculate them once
        resultsSave = {
            'X_train':x_train1,
            'X_train_ori':X_train_ori,
            'X_test':x_test,
            'X_test_ori':X_test_ori,
            'X_val': x_val,
            'X_val_ori':X_val_ori,
            'y_trainy':y_trainy,
            'y_train':y_train1,
            'y_val': y_val,
            'y_test':y_test,
            'y_testy':y_testy
        }
        save_obj(resultsSave, processedDataName)
    return x_train1, x_val, x_test, y_train1, y_val, y_test, X_train_ori, X_val_ori, X_test_ori, y_trainy, y_testy



def getHInceptTimeResults():
    results = {
        "ACSF1":0.94,
        "Adiac":0.843989769820972,
        "AllGestureWiimoteX":0.791428571428571,
        "AllGestureWiimoteY":0.835714285714286,
        "AllGestureWiimoteZ":0.802857142857143,
        "ArrowHead":0.891428571428571,
        "BME":0.993333333333333,
        "Beef":0.7,
        "BeetleFly":0.75,
        "BirdChicken":0.95,
        "CBF":0.997777777777778,
        "Car":0.9,
        "Chinatown":0.979591836734694,
        "ChlorineConcentration":0.882552083333333,
        "CinCECGTorso":0.844202898550725,
        "Coffee":1,
        "Computers":0.812,
        "CricketX":0.853846153846154,
        "CricketY":0.851282051282051,
        "CricketZ":0.853846153846154,
        "Crop":0.772142857142857,
        "DiatomSizeReduction":0.947712418300653,
        "DistalPhalanxOutlineAgeGroup":0.741007194244604,
        "DistalPhalanxOutlineCorrect":0.797101449275362,
        "DistalPhalanxTW":0.654676258992806,
        "DodgerLoopDay":0.5875,
        "DodgerLoopGame":0.833333333333333,
        "DodgerLoopWeekend":0.971014492753623,
        "ECG200":0.9,
        "ECG5000":0.939333333333333,
        "ECGFiveDays":1,
        "EOGHorizontalSignal":0.621546961325967,
        "EOGVerticalSignal":0.466850828729282,
        "Earthquakes":0.726618705035971,
        "ElectricDevices":0.714044870963559,
        "EthanolLevel":0.788,
        "FaceAll":0.818343195266272,
        "FaceFour":0.954545454545455,
        "FacesUCR":0.96780487804878,
        "FiftyWords":0.848351648351648,
        "Fish":0.982857142857143,
        "FordA":0.961363636363636,
        "FordB":0.853086419753086,
        "FreezerRegularTrain":0.99719298245614,
        "FreezerSmallTrain":0.822456140350877,
        "Fungi":1,
        "GestureMidAirD1":0.753846153846154,
        "GestureMidAirD2":0.730769230769231,
        "GestureMidAirD3":0.423076923076923,
        "GesturePebbleZ1":0.930232558139535,
        "GesturePebbleZ2":0.886075949367089,
        "GunPoint":1,
        "GunPointAgeSpan":0.996835443037975,
        "GunPointMaleVersusFemale":1,
        "GunPointOldVersusYoung":0.977777777777778,
        "Ham":0.704761904761905,
        "HandOutlines":0.954054054054054,
        "Haptics":0.568181818181818,
        "Herring":0.71875,
        "HouseTwenty":0.974789915966386,
        "InlineSkate":0.525454545454546,
        "InsectEPGRegularTrain":1,
        "InsectEPGSmallTrain":0.91566265060241,
        "InsectWingbeatSound":0.644444444444444,
        "ItalyPowerDemand":0.968901846452867,
        "LargeKitchenAppliances":0.896,
        "Lightning2":0.80327868852459,
        "Lightning7":0.849315068493151,
        "Mallat":0.964179104477612,
        "Meat":0.95,
        "MedicalImages":0.801315789473684,
        "MelbournePedestrian":0.911849118491185,
        "MiddlePhalanxOutlineAgeGroup":0.519480519480519,
        "MiddlePhalanxOutlineCorrect":0.838487972508591,
        "MiddlePhalanxTW":0.512987012987013,
        "MixedShapesRegularTrain":0.976494845360825,
        "MixedShapesSmallTrain":0.921649484536082,
        "MoteStrain":0.88258785942492,
        "NonInvasiveFetalECGThorax1":0.960814249363868,
        "NonInvasiveFetalECGThorax2":0.966921119592875,
        "OSULeaf":0.950413223140496,
        "OliveOil":0.766666666666667,
        "PLAID":0.942271880819367,
        "PhalangesOutlinesCorrect":0.842657342657343,
        "Phoneme":0.338080168776371,
        "PickupGestureWiimoteZ":0.78,
        "PigAirwayPressure":0.567307692307692,
        "PigArtPressure":0.995192307692308,
        "PigCVP":0.966346153846154,
        "Plane":1,
        "PowerCons":0.938888888888889,
        "ProximalPhalanxOutlineAgeGroup":0.839024390243902,
        "ProximalPhalanxOutlineCorrect":0.924398625429553,
        "ProximalPhalanxTW":0.780487804878049,
        "RefrigerationDevices":0.52,
        "Rock":0.84,
        "ScreenType":0.576,
        "SemgHandGenderCh2":0.823333333333333,
        "SemgHandMovementCh2":0.531111111111111,
        "SemgHandSubjectCh2":0.844444444444444,
        "ShakeGestureWiimoteZ":0.92,
        "ShapeletSim":1,
        "ShapesAll":0.928333333333333,
        "SmallKitchenAppliances":0.776,
        "SmoothSubspace":0.973333333333333,
        "SonyAIBORobotSurface1":0.848585690515807,
        "SonyAIBORobotSurface2":0.947534102833158,
        "StarLightCurves":0.975594949004371,
        "Strawberry":0.983783783783784,
        "SwedishLeaf":0.9728,
        "Symbols":0.979899497487437,
        "SyntheticControl":0.996666666666667,
        "ToeSegmentation1":0.960526315789473,
        "ToeSegmentation2":0.930769230769231,
        "Trace":1,
        "TwoLeadECG":0.997366110623354,
        "TwoPatterns":1,
        "UMD":0.993055555555556,
        "UWaveGestureLibraryAll":0.947794528196538,
        "UWaveGestureLibraryX":0.830820770519263,
        "UWaveGestureLibraryY":0.782802903405919,
        "UWaveGestureLibraryZ":0.785315466219989,
        "Wafer":0.999188838416612,
        "Wine":0.648148148148148,
        "WordSynonyms":0.758620689655172,
        "Worms":0.831168831168831,
        "WormsTwoClass":0.779220779220779,
        "Yoga":0.928
    }
    return results
