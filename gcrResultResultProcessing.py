#!/usr/bin/env python
# coding: utf-8
##test

# In[2]:


import os
import numpy as np

import pandas as pd

from modules import helper
from scipy import stats
from scipy.stats.stats import pearsonr
from modules import saliencyHelper as sh

#from sacred import Experiment
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from modules import dataset_selecter as ds
from modules import pytorchTrain as pt
from modules import saliencyHelper as sh
from modules import helper
from modules import cdd
from modules import GCRPlus
from sklearn import metrics
from scipy.stats import rankdata
from operator import itemgetter

import matplotlib.colors as mcolors

from pyts.datasets import ucr_dataset_list
from pyts.datasets import ucr_dataset_info


# 

# In[3]:


class config:
    def __init__(self, configName, hyperss, hypNames, test_sizes, topLevelss, toplevels, dataset, symbols, nrEmpty,andStack,orStack,xorStack,nrAnds,nrOrs,nrxor,trueIndexes,orOffSet,xorOffSet,redaundantIndexes,batch_size):
        self.configName=configName
        self.hyperss=hyperss
        self.hypNames=hypNames
        self.test_sizes=test_sizes
        self.topLevelss=topLevelss
        self.toplevels=toplevels
        self.dataset=dataset
        self.symbols=symbols
        self.nrEmpty=nrEmpty
        self.andStack=andStack
        self.orStack=orStack
        self.xorStack=xorStack
        self.nrAnds=nrAnds
        self.nrOrs=nrOrs
        self.nrxor=nrxor
        self.trueIndexes=trueIndexes
        self.orOffSet=orOffSet
        self.xorOffSet=xorOffSet
        self.redaundantIndexes=redaundantIndexes
        self.batch_size=batch_size

allConfigs = []
models = [['Transformer', 500, 64, 0.5, True, True, 4 ,  3, 0 , 0, False, 'Transformer L3'] 
      ,['Transformer', 500, 32, 0.5, True, True, 4 ,  2, 0 , 0, False, 'Transformer L2']
      ,['CNN', 500, 16, 1, True , True , 0 ,  3, 0 , 0, False, 'ResNet L3'] 
      , ['CNN', 500, 16, 1, True , True , 0 ,  2, 0 , 0, False, 'ResNet L2']
    ]


folderGeneral = './BilderGCR/general/'
if not os.path.exists(folderGeneral):
    os.makedirs(folderGeneral)


''# In[4]:


accTreshold = 1 



''# In[5]:


def arrayToString(indexes):
    out = ""
    for i in indexes:
        out = out + ',' + str(i)
    return out



def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')


# In[ ]:





# In[6]:

# In[7]:


filteredResultsFolder = ['filteredSymbolicGCR']

kNames = ['Rollout', 'Attention', 'IntegratedGradients', 'FeaturePermutation', 'Random']
ks = ['LRP-rollout', 'Attention-.', 'captum-IntegratedGradients', 'captum-FeaturePermutation', 'Random-Random']
modesPerK = [0,0,0,0,0]

kIter = ks
modes = [0]

allReses = dict()
for rFolder in filteredResultsFolder:
    if rFolder not in allReses.keys():
        allReses[rFolder] = dict()

dataset_list = ds.univariate_equal_length 

# In[ ]:


normNames = GCRPlus.getAllNeededReductionNames()

fcamOptions = ['rMA', 'rMS']
fcamONames = ['r.avg.', 'sum']
gtmOptions = GCRPlus.gtmReductionStrings()
gcrNames = ['2nd-Order', '1st-Order']


# In[8]:

numberFeaturesSet = [200]


fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")

allRes = dict()

assemblyPredictions = []
symbolificationStrategySet = ['uniform']

assemblyAcc = []
bestParams = dict() 
bestParams['model acc'] = []
bestParams['RandomForestClassifier acc']  = []
bestParams['AdaBoostClassifier acc'] = []
bestParams['RotationForest acc'] = []
bestParams['assembly acc'] = []
bestParams['oldSetting acc'] = []
bestParams2 = dict() 
bestParams2['model acc'] = []
bestParams2['RandomForestClassifier acc']  = []
bestParams2['AdaBoostClassifier acc'] = []
bestParams2['RotationForest acc'] = []
bestParams2['assembly acc'] = []
bestParams2['oldSetting acc'] = []
oldTSFreshSelSym = [] 
ourSahredDataSets = [] 



assemblyPerformance = []
for di in range(112):
    assemblyPerformance.append([])
    for symbolsCount in [5,9]:
        for numberFeatures in numberFeaturesSet:
            for symbolificationStrategy in symbolificationStrategySet:
                for j, modelSelection in enumerate([models]):
                    for ci, hypers in enumerate(modelSelection):

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
                        configName = hypers[-1]

                        
                    

                        dataName = dataset_list[di]
                                
                        for rFolder in filteredResultsFolder:

                            dsName = str(dataName) +  '-n:' + str("UCR")
                            saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                            if os.path.isfile(saveName + '.pkl'):
                                print('FOUND: ' + saveName)
                                if saveName in allReses[rFolder].keys():
                                    res = allReses[rFolder][saveName]
                                else:
                                    try:
                                        results = helper.load_obj(saveName)
                                    except Exception as e:
                                        print(e)
                                        continue

                                    res = dict()
                                    for index, vv in np.ndenumerate(results):
                                        res = vv

                                    allReses[rFolder][saveName] = res
                            else:
                                print('NOT FOUND: ' + saveName)
                                continue

                            assemblyPerformance[-1].append(res['assembly predictions'])


for di in range(112):
    if len(assemblyPerformance[di]) > 0:
        _, _, _, y_test, _, y_testy, _, _, dataName, _ = ds.datasetSelector('UCR', 42, number=di)

        testPredy = np.argmax(np.mean(np.array(assemblyPerformance[di]), axis=0), axis=1)
        assemblyAcc.append(metrics.accuracy_score(testPredy, np.argmax(y_test,axis=1)))
        print(assemblyAcc[di])
        f = open(folderGeneral+ "fullAssemblyPerformance.txt", "a")
        f.write(str(dataName) + ' ' + str(di) + ', ' + str(assemblyAcc[-1]))
        f.close()
    else:
        assemblyAcc.append(np.nan)



df = pd.read_csv('best_config_per_dataset.csv')

oldDatasetIndexes = []

columnCounts = dict()
for di in range(112):
    bestParams['model acc'].append([])
    bestParams['RandomForestClassifier acc'].append([])
    bestParams['AdaBoostClassifier acc'].append([])
    bestParams['RotationForest acc'].append([])
    bestParams['assembly acc'].append([])
    bestParams['oldSetting acc'].append([])
    dataName = dataset_list[di]


    bestParams2['model acc'].append([])
    bestParams2['RandomForestClassifier acc'].append([])
    bestParams2['AdaBoostClassifier acc'].append([])
    bestParams2['RotationForest acc'].append([])
    bestParams2['assembly acc'].append([])
    bestParams2['oldSetting acc'].append([])

    for dvi,dv in enumerate(df[df['name'] == dataName]['Test Accuracy']):
        oldDatasetIndexes.append(di)
        break

    dataName = dataset_list[di]
    for numberFeatures in numberFeaturesSet:
        
        for symbolificationStrategy in symbolificationStrategySet:
            for symbolsCount in [5,9]:
                
                for rFolder in filteredResultsFolder:

                    wname = pt.getDatasetName('UCR', di, numberFeatures,  'MCB', symbolificationStrategy, True, True, symbolsCount, 5, 42, resultsPath = "preprocessingSymbolicGCR2")

                    
                    if os.path.isfile(wname + '.pkl'):
                        print('FOUND: ' + wname)

                        try:
                            results = helper.load_obj(wname)
                        except Exception as e:
                            print(e)
                            continue
                        preProcessRes = dict()
                        for index, vv in np.ndenumerate(results):
                            preProcessRes = vv
                        
                        for f in preProcessRes['columns']:
                            for c in f:
                                if c in columnCounts.keys():
                                    columnCounts[c] += 1
                                else:
                                    columnCounts[c] = 1

                    else:
                        print('NOT FOUND: ' + wname)


f = open(folderGeneral+ "columSelection.txt", "a")
for k,v in sorted(columnCounts.items(), key=lambda p:p[1], reverse=True):
    f.write(k + ' ' + str(v) + '\n')
f.close()

ndf = pd.DataFrame([], columns=['classifier_name', 'domain', 'accuracy'])
pdf = pd.DataFrame([], columns=['classifier_name', 'dataset_name', 'accuracy'])
accs = pd.DataFrame([], columns=['method', 'Test Accuracy'])
inputIndex = 0
avgAccsIndex = 0

df = pd.read_csv('best_config_per_dataset.csv')

ax1 = ax
width = 0
n_bars = 9
standardWidth = 0.9
bar_width = 0.9 / n_bars
barInd = 0
rects = []
lNames = []
firstAdd = True
for symbolsCount in [5,9]: 
    for numberFeatures in numberFeaturesSet: 
        
        for j, modelSelection in enumerate([models]):
            for ci, hypers in enumerate(modelSelection):

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
                configName = hypers[-1]

                lNames.append('symbols ' +str(symbolsCount) + ', features ' + str(numberFeatures) + ', ' + modelType + ' l' +str(numOfLayers))


                x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                barInd+= 1
                width += standardWidth
                resultVs = [[],[],[],[],[],[],[]]

                symbolificationStrategy = symbolificationStrategySet[0] 

                lables = ['Our Model', 'Random Forest', 'AdaBoost', 'Rotational Forest', 'Assembly', 'Full Assembly', 'H-InceptionTime'] 


                for di in range(112):

                    dataName = dataset_list[di]
                            
                    for rFolder in filteredResultsFolder:

                        dsName = str(dataName) +  '-n:' + str("UCR")
                        saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results=True, resultsPath=rFolder)

                        if os.path.isfile(saveName + '.pkl'):
                            print('FOUND: ' + saveName)
                            if saveName in allReses[rFolder].keys():
                                res = allReses[rFolder][saveName]
                            else:
                                try:
                                    results = helper.load_obj(saveName)
                                except Exception as e:
                                    print(e)
                                    bestParams['model acc'][di].append(np.nan)
                                    bestParams['RandomForestClassifier acc'][di].append(np.nan)
                                    bestParams['AdaBoostClassifier acc'][di].append(np.nan)
                                    bestParams['RotationForest acc'][di].append(np.nan)
                                    bestParams['assembly acc'][di].append(np.nan)
                                    bestParams['oldSetting acc'][di].append(np.nan)
                                    continue

                                res = dict()
                                for index, vv in np.ndenumerate(results):
                                    res = vv

                                allReses[rFolder][saveName] = res
                        else:
                            print('NOT FOUND: ' + saveName)
                            bestParams['model acc'][di].append(np.nan)
                            bestParams['RandomForestClassifier acc'][di].append(np.nan)
                            bestParams['AdaBoostClassifier acc'][di].append(np.nan)
                            bestParams['RotationForest acc'][di].append(np.nan)
                            bestParams['assembly acc'][di].append(np.nan)
                            bestParams['oldSetting acc'][di].append(np.nan)
                            continue

                        for fi, f in enumerate(res['model acc']):
                            resultVs[0].append(res['model acc'][fi])

                            resultVs[1].append(res['RandomForestClassifier acc'][fi])

                            resultVs[2].append(res['AdaBoostClassifier acc'][fi])

                            resultVs[3].append(res['RotationForest acc'][fi])

                        resultVs[4].append(res['assembly acc'])
                        resultVs[5].append(assemblyAcc[di])
                        resultVs[6].append(helper.getHInceptTimeResults()[dataName])

                        bestParams['model acc'][di].append(np.mean(res['model acc']))
                        
                        pdf.loc[inputIndex] = [lables[0] + ' ' + lNames[-1], dataName,np.mean(res['model acc'])]
                        inputIndex +=1
                        bestParams['RandomForestClassifier acc'][di].append(np.mean(res['RandomForestClassifier acc']))
                        if firstAdd:
                            pdf.loc[inputIndex] = [lables[1],dataName,np.mean(res['RandomForestClassifier acc'])]
                            inputIndex +=1
                        bestParams['AdaBoostClassifier acc'][di].append(np.mean(res['AdaBoostClassifier acc']))
                        if firstAdd:
                            pdf.loc[inputIndex] = [lables[2],dataName,np.mean(res['AdaBoostClassifier acc'])]
                            inputIndex +=1
                        bestParams['RotationForest acc'][di].append(np.mean(res['RotationForest acc']))
                        if firstAdd:
                            pdf.loc[inputIndex] = [lables[3] ,dataName,np.mean(res['RotationForest acc'])]
                            inputIndex +=1
                        bestParams['assembly acc'][di].append(res['assembly acc'])
                        pdf.loc[inputIndex] = [lables[4] + ' ' + lNames[-1],dataName,res['assembly acc']]
                        inputIndex +=1

                firstAdd = False   
                resultVstd = []
                for vi, v in enumerate(resultVs):
                    print(vi)
                    print(v)
                    resultVs[vi] = np.nanmean(v)
                    resultVstd.append(np.nanstd(v))

                print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))

                counts = np.round(resultVs,4)
                e =  np.round(resultVstd,4)




                ind = np.arange(len(counts))
                

                rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                rects.append(rect)
                ax.set_xticks(ind)
                ax.set_xticklabels(lables, fontsize=10)
                ax.set_ylabel('Percent', fontsize=10)
                ax.set_title('Model Performance')
                ax.set_ylim(bottom=0, top=1)
                
                ax.tick_params(labelrotation=90)


domainIndex = 0

domains = {
    'CinCECGTorso': 'ECG',
    'MixedShapesRegularTrain' : 'IMAGE',
    'StarLightCurves': 'SENSOR'
}


for di in range(112):
    dataName = dataset_list[di]
    if not np.isnan(assemblyAcc[di]):
        pdf.loc[inputIndex] = ['Full Assembly',dataName,assemblyAcc[di]]
        inputIndex +=1
        pdf.loc[inputIndex] = ['H-InceptionTime',dataName,helper.getHInceptTimeResults()[dataName]]
        inputIndex +=1
        if helper.getHInceptTimeResults()[dataName] > assemblyAcc[di]:
            f = open(folderGeneral+ "worseThanHInception.txt", "a")
            f.write(dataName + ' H-Time: ' + str(helper.getHInceptTimeResults()[dataName]) + ' Ours: ' + str(assemblyAcc[di]) + ' Fundaments: ' +  str(bestParams['model acc'][di]) + '\ns')
            f.close()
            
        
        if dataName in domains.keys():
            utype = domains[dataName]
        else:
            utype = ucr_dataset_info(dataName)['type']
        ndf.loc[domainIndex] = ['Full Assembly',utype.upper(), assemblyAcc[di]]
        domainIndex += 1
        ndf.loc[domainIndex] = ['H-InceptionTime',utype.upper(),helper.getHInceptTimeResults()[dataName]]
        domainIndex += 1
        ndf.loc[domainIndex] = ['Coverage',utype.upper(), assemblyAcc[di] >= helper.getHInceptTimeResults()[dataName]]
        domainIndex += 1
        

#TODO count?
ndf = ndf.groupby(['classifier_name', 'domain'])['accuracy'].mean().reset_index()
ndf.to_csv(folderGeneral+'domainPerformance.csv')


lNames.append('best config')



x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
barInd+= 1
width += standardWidth
resultVs = [[],[],[],[],[],[],[]]
resultVs[0].append(np.max(np.array(bestParams['model acc']), axis=1))
resultVs[1].append(np.max(bestParams['RandomForestClassifier acc'], axis=1))
resultVs[2].append(np.max(bestParams['AdaBoostClassifier acc'], axis=1))
resultVs[3].append(np.max(bestParams['RotationForest acc'], axis=1))
resultVs[4].append(np.max(bestParams['assembly acc'], axis=1))
resultVs[5].append(0)
resultVs[6].append(0)




resultVstd = []
for vi, v in enumerate(resultVs):
    print(vi)
    print(v)
    resultVs[vi] = np.nanmean(v)
    resultVstd.append(np.nanstd(v))
print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))

lables = ['Our Model', 'Random Forest', 'AdaBoost', 'Rotational Forest', 'Assembly', 'Full Assembly', 'H-InceptionTime']  #'Full Assembly Acc',
counts = np.round(resultVs,4)
e =  np.round(resultVstd,4)


ind = np.arange(len(counts))


rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
rects.append(rect)
ax.set_xticks(ind)
ax.set_xticklabels(lables, fontsize=10)
ax.set_ylabel('Percent', fontsize=10)
ax.set_title('Model Performance')
ax.set_ylim(bottom=0, top=1)

ax.tick_params(labelrotation=90)

plt.rcParams.update({'font.size': 18})
specificFolder = folderGeneral + 'performance/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
fig.legend(rects, labels=lNames, 
                loc="upper right", bbox_to_anchor=(1.488, 0.99)) 
fig.tight_layout()

fig.savefig(specificFolder + 'generalPerformances.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()





pdfM = pdf[['classifier_name', 'accuracy']].groupby(['classifier_name'])
print(pdfM)
for cn, a in pdfM:
    accs.loc[avgAccsIndex] = [cn[0], np.mean(a['accuracy'].to_numpy())]
    avgAccsIndex+= 1

pdf.to_csv(folderGeneral+'fullPerformance.csv')
pdf = pdf.sort_values(by=['dataset_name'])
accs.to_csv(folderGeneral+'avgPerformance.csv')


cdd.draw_cd_diagram(df_perf=pdf, accs=accs, folder=folderGeneral, title='Accuracy', labels=True)

    



fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")

pdf = pd.DataFrame([], columns=['classifier_name', 'dataset_name', 'accuracy'])
accs = pd.DataFrame([], columns=['method', 'Test Accuracy'])
inputIndex = 0
avgAccsIndex = 0

ax1 = ax
width = 0
n_bars = 9
standardWidth = 0.9
bar_width = 0.9 / n_bars
barInd = 0
rects = []
lNames = []
firstRun = True

for symbolsCount in [5,9]:
    for numberFeatures in numberFeaturesSet:
        
        for j, modelSelection in enumerate([models]):
            for ci, hypers in enumerate(modelSelection):

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
                configName = hypers[-1]

                lNames.append('symbols ' +str(symbolsCount) + ', features ' + str(numberFeatures) + ', ' + modelType + ' l' +str(numOfLayers))


                x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                barInd+= 1
                width += standardWidth
                resultVs = [[],[],[],[],[],[],[],[]]

                symbolificationStrategy = symbolificationStrategySet[0] 

                lables = ['Our Model (New)', 'Random Forest', 'AdaBoost', 'Rotational Forest', 'Assembly', 'Full Assembly', 'H-InceptionTime', 'Our Model (Old)'] 


                for di in range(112):

                    dataName = dataset_list[di]
                            
                    for rFolder in filteredResultsFolder:

                        dsName = str(dataName) +  '-n:' + str("UCR")

                        added = False
                        for dvi,dv in enumerate(df[df['name'] == dataName]['Test Accuracy']):
                            added = True
                            resultVs[7].append(dv)

                        saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results=True, resultsPath=rFolder)

                        if not added:
                            continue

                        if os.path.isfile(saveName + '.pkl'):
                            print('FOUND: ' + saveName)
                            if saveName in allReses[rFolder].keys():
                                res = allReses[rFolder][saveName]
                            else:
                                try:
                                    results = helper.load_obj(saveName)
                                except Exception as e:
                                    print(e)
                                    bestParams2['model acc'][di].append(np.nan)
                                    bestParams2['RandomForestClassifier acc'][di].append(np.nan)
                                    bestParams2['AdaBoostClassifier acc'][di].append(np.nan)
                                    bestParams2['RotationForest acc'][di].append(np.nan)
                                    bestParams2['assembly acc'][di].append(np.nan)
                                    bestParams2['oldSetting acc'][di].append(np.nan)
                                    continue

                                res = dict()
                                for index, vv in np.ndenumerate(results):
                                    res = vv

                                allReses[rFolder][saveName] = res
                        else:
                            print('NOT FOUND: ' + saveName)
                            bestParams2['model acc'][di].append(np.nan)
                            bestParams2['RandomForestClassifier acc'][di].append(np.nan)
                            bestParams2['AdaBoostClassifier acc'][di].append(np.nan)
                            bestParams2['RotationForest acc'][di].append(np.nan)
                            bestParams2['assembly acc'][di].append(np.nan)
                            bestParams2['oldSetting acc'][di].append(np.nan)
                            continue

                        for fi, f in enumerate(res['model acc']):
                            resultVs[0].append(res['model acc'][fi])

                            resultVs[1].append(res['RandomForestClassifier acc'][fi])

                            resultVs[2].append(res['AdaBoostClassifier acc'][fi])


                            resultVs[3].append(res['RotationForest acc'][fi])

                        resultVs[4].append(res['assembly acc'])
                        resultVs[5].append(assemblyAcc[di])
                        resultVs[6].append(helper.getHInceptTimeResults()[dataName])



                        bestParams2['model acc'][di].append(np.mean(res['model acc']))
                        

                        if pdf.query("classifier_name == 'New Model Selection' and dataset_name == @dataName").size >= 1:
                            ind = pdf.query("classifier_name == 'New Model Selection' and dataset_name == @dataName")
                            if ind['accuracy'].iloc[0] < np.mean(res['model acc']):
                                ind['accuracy'].iloc[0] = np.mean(res['model acc'])
                                pdf.loc[(pdf.classifier_name == 'New Model Selection') & (pdf.dataset_name == dataName),'accuracy'] = np.mean(res['model acc'])

                        else:
                            pdf.loc[inputIndex] = ['New Model Selection', dataName,np.mean(res['model acc'])]
                            inputIndex +=1
                        bestParams2['RandomForestClassifier acc'][di].append(np.mean(res['RandomForestClassifier acc']))
                        if firstRun:
                            pdf.loc[inputIndex] = [lables[1],dataName,np.mean(res['RandomForestClassifier acc'])]
                            inputIndex +=1
                        bestParams2['AdaBoostClassifier acc'][di].append(np.mean(res['AdaBoostClassifier acc']))
                        if firstRun:
                            pdf.loc[inputIndex] = [lables[2],dataName,np.mean(res['AdaBoostClassifier acc'])]
                            inputIndex +=1
                        bestParams2['RotationForest acc'][di].append(np.mean(res['RotationForest acc']))
                        if firstRun:
                            pdf.loc[inputIndex] = [lables[3],dataName,np.mean(res['RotationForest acc'])]
                            inputIndex +=1
                        bestParams2['assembly acc'][di].append(res['assembly acc'])
                        pdf.loc[inputIndex] = [lables[4] + ' ' + lNames[-1],dataName,res['assembly acc']]
                        inputIndex +=1


                        if firstRun and not np.isnan(assemblyAcc[di]):
                            pdf.loc[inputIndex] = ['Full Assembly',dataName,assemblyAcc[di]]
                            inputIndex +=1
                            pdf.loc[inputIndex] = ['H-InceptionTime',dataName,helper.getHInceptTimeResults()[dataName]]
                            inputIndex +=1
                            pdf.loc[inputIndex] = ["Old Model Selection" ,dataName,np.mean(df[df['name'] == dataName]['Test Accuracy'])]
                            inputIndex +=1

                firstRun = False        
                resultVstd = []
                for vi, v in enumerate(resultVs):
                    print(vi)
                    print(v)
                    resultVs[vi] = np.nanmean(v)
                    resultVstd.append(np.nanstd(v))

                print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))

                counts = np.round(resultVs,4)
                e =  np.round(resultVstd,4)



                ind = np.arange(len(counts))
                

                rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                rects.append(rect)
                ax.set_xticks(ind)
                ax.set_xticklabels(lables, fontsize=10)
                ax.set_ylabel('Percent', fontsize=10)
                ax.set_title('Model Performance')
                ax.set_ylim(bottom=0, top=1)
                
                ax.tick_params(labelrotation=90)



lNames.append('best config')



x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
barInd+= 1
width += standardWidth
resultVs = [[],[],[],[],[],[],[],[]]

print('---')
print(bestParams2['model acc'])
oldDatasetIndexes = np.array(oldDatasetIndexes)
print(oldDatasetIndexes)
print(np.array(np.array(bestParams2['model acc'])[oldDatasetIndexes].tolist()))


resultVs[0].append(np.max(np.array(np.array(bestParams2['model acc'])[oldDatasetIndexes].tolist()), axis=1))
resultVs[1].append(np.max(np.array(np.array(bestParams2['RandomForestClassifier acc'])[oldDatasetIndexes].tolist()), axis=1))
resultVs[2].append(np.max(np.array(np.array(bestParams2['AdaBoostClassifier acc'])[oldDatasetIndexes].tolist()), axis=1))
resultVs[3].append(np.max(np.array(np.array(bestParams2['RotationForest acc'])[oldDatasetIndexes].tolist()), axis=1))
resultVs[4].append(np.max(np.array(np.array(bestParams2['assembly acc'])[oldDatasetIndexes].tolist()), axis=1))
resultVs[5].append(0)
resultVs[6].append(0)
resultVs[7].append(0)




resultVstd = []
for vi, v in enumerate(resultVs):
    print(vi)
    print(v)
    resultVs[vi] = np.nanmean(v)
    resultVstd.append(np.nanstd(v))
print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))

lables = ['Our Model (New)', 'Random Forest', 'AdaBoost', 'Rotational Forest', 'Assembly', 'Full Assembly', 'H-InceptionTime', 'Our Model (Old)']  #'Full Assembly Acc',
counts = np.round(resultVs,4)
e =  np.round(resultVstd,4)


ind = np.arange(len(counts))


rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
rects.append(rect)
ax.set_xticks(ind)
ax.set_xticklabels(lables, fontsize=10)
ax.set_ylabel('Percent', fontsize=10)
ax.set_title('Model Performance')
ax.set_ylim(bottom=0, top=1)

ax.tick_params(labelrotation=90)

plt.rcParams.update({'font.size': 18})
specificFolder = folderGeneral + 'performance/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
fig.legend(rects, labels=lNames, 
                loc="upper right", bbox_to_anchor=(1.488, 0.99)) 
fig.tight_layout()

fig.savefig(specificFolder + 'generalPerformancesOld.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()

pdfM = pdf[['classifier_name', 'accuracy']].groupby(['classifier_name'])
print(pdfM)
for cn, a in pdfM:
    accs.loc[avgAccsIndex] = [cn[0], np.mean(a['accuracy'].to_numpy())]
    avgAccsIndex+= 1
    
pdf = pdf.groupby(['classifier_name', 'dataset_name'])['accuracy'].mean().reset_index()



cdd.draw_cd_diagram(df_perf=pdf, accs=accs, folder=specificFolder, title='Accuracy Old', labels=True)


# In[8]:

import sys


def getBestMode(combis, k, gcrOption, normName, metric, useMax=True):
    modeVs = []
    for m in modesResults.keys():
        modeVs.append([[],[],[],[],[],[],[]])
        
        for vi, v in modesResults[m]:

            modeVs[m][vi].append(np.nanmean(v))
    
    if useMax:
        return list(modesResults.keys())[np.argmax(np.array(modeVs), axis=0)]
    else:
        return list(modesResults.keys())[np.argmin(np.array(modeVs), axis=0)]



colorN = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple']
fig, axs = plt.subplots(ncols=len(fcamOptions) + len(gtmOptions), sharex=True, sharey=True,
                                figsize=(28, 12), layout="constrained")


jCounter = -1
for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]):
    for gi, gcrOption in enumerate(gcrOptions):
        jCounter+=1
        ax = axs[jCounter]
        ax.set_title('SimpleGCR ' + fcamONames[gi])
        
        
        width = 0
        n_bars = 4
        standardWidth = 0.8
        bar_width = 0.8 / n_bars
        barInd = 0
        rects = []
        llables = []
        for j, modelSelection in enumerate([models]):
            for ci, hypers in enumerate(modelSelection):
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
                configName = hypers[-1]

                llables.append(configName)


                for normName in normNames:
                    resultVs = [[],[],[],[],[]]
                    x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                    barInd+= 1
                    width += standardWidth
                    
                    for di in range(112):
                        for symbolsCount in [5,9]:
                            for numberFeatures in numberFeaturesSet:
                                for symbolificationStrategy in symbolificationStrategySet:

                                    dataName = dataset_list[di]
                                            
                                    for rFolder in filteredResultsFolder:

                                        dsName = str(dataName) +  '-n:' + str("UCR")
                                        saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                                        if os.path.isfile(saveName + '.pkl'):
                                            print('FOUND: ' + saveName)
                                            if saveName in allReses[rFolder].keys():
                                                res = allReses[rFolder][saveName]
                                            else:
                                                try:
                                                    results = helper.load_obj(saveName)
                                                except Exception as e:
                                                    print(e)
                                                    continue

                                                res = dict()
                                                for index, vv in np.ndenumerate(results):
                                                    res = vv

                                                allReses[rFolder][saveName] = res
                                        else:
                                            print('NOT FOUND: ' + saveName)
                                            continue




                                        for fi, f in enumerate(res['simpleGCR'][gcrOption][normName]['acc']):

                                            resultVs[0].append(res['simpleGCR'][gcrOption][normName]['acc'][fi])
                                            resultVs[1].append(res['simpleGCR'][gcrOption][normName]['predicsion'][fi])
                                            resultVs[2].append(res['simpleGCR'][gcrOption][normName]['recall'][fi])
                                            resultVs[3].append(res['simpleGCR'][gcrOption][normName]['f1'][fi])
                                            resultVs[4].append(res['simpleGCR'][gcrOption][normName]['giniScore'][fi])

                    resultVstd = []
                    for vi, v in enumerate(resultVs):
                        resultVs[vi] = np.mean(v)
                        resultVstd.append(np.std(v))
                    print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))
                    
                    lables = ['Acc.', 'Prec.', 'Rec.', 'F1', 'Gini']
                    counts = np.round(resultVs,4)
                    e =  np.round(resultVstd,4)

                    ind = np.arange(len(counts))
                    rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                    rects.append(rect)
                    ax.set_xticks(ind)
                    ax.set_xticklabels(lables, fontsize=10)
                    ax.set_ylabel('Accuracy', fontsize=10)
                    ax.set_ylim(bottom=0, top=1)
                    
                    ax.tick_params(labelrotation=90)

specificFolder = folderGeneral + 'simpleGCR/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
legend =llables
fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.088, 0.85)) 
fig.tight_layout()

fig.savefig(specificFolder + 'simpleGCRPerformance.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()


for ki, k in enumerate(ks):
    fig, axs = plt.subplots(ncols=len(fcamOptions) + len(gtmOptions), sharex=True, sharey=True,
                                    figsize=(28, 12), layout="constrained")


    jCounter = -1
    for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]):
        for gi, gcrOption in enumerate(gcrOptions):
            jCounter+=1
            ax = axs[jCounter]
            ax.set_title(kNames[ki] + ' ' + fcamONames[gi])

            width = 0
            n_bars = 4
            standardWidth = 0.8
            bar_width = 0.8 / n_bars
            barInd = 0
            rects = []

            lnames = []

            for j, modelSelection in enumerate([models]):
                for ci, hypers in enumerate(modelSelection):
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
                    configName = hypers[-1]

                    lnames.append(configName)
                    
                    
                    for normName in normNames:
                        resultVs = np.zeros(6)
                        x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                        barInd+= 1
                        width += standardWidth


                        modesResults = dict()
                        for mode in modes:
                            modesResults[mode] = [[],[],[],[],[],[]]
                            for di in range(112):
                                for symbolsCount in [5,9]:
                                    for numberFeatures in numberFeaturesSet:
                                        for symbolificationStrategy in symbolificationStrategySet:
                                            dataName = dataset_list[di]
                                                    
                                            for rFolder in filteredResultsFolder:

                                                dsName = str(dataName) +  '-n:' + str("UCR")
                                                saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                                                if os.path.isfile(saveName + '.pkl'):
                                                    print('FOUND: ' + saveName)
                                                    if saveName in allReses[rFolder].keys():
                                                        res = allReses[rFolder][saveName]
                                                    else:
                                                        try:
                                                            results = helper.load_obj(saveName)
                                                        except Exception as e:
                                                            print(e)
                                                            continue

                                                        res = dict()
                                                        for index, vv in np.ndenumerate(results):
                                                            res = vv

                                                        allReses[rFolder][saveName] = res
                                                else:
                                                    print('NOT FOUND: ' + saveName)
                                                    continue



                                                if k in res['gcr'].keys():
                                                    for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName]['acc']):
                                                        m = []
                                                        modesResults[mode][0].append(res['gcr'][k][mode][gcrOption][normName]['acc'][fi])
                                                        modesResults[mode][1].append(res['gcr'][k][mode][gcrOption][normName]['predicsion'][fi])
                                                        modesResults[mode][2].append(res['gcr'][k][mode][gcrOption][normName]['recall'][fi])
                                                        modesResults[mode][3].append(res['gcr'][k][mode][gcrOption][normName]['f1'][fi])
                                                        modesResults[mode][4].append(res['gcr'][k][mode][gcrOption][normName]['giniScore'][fi])

                                                        modesResults[mode][5].append(metrics.auc(range(len(res['gcr'][k][mode][gcrOption][normName]['confidence'][fi])), res['gcr'][k][mode][gcrOption][normName]['confidence'][fi]) / 10)


                        lables = ['Acc.', 'Prec.', 'Rec.', 'F1', 'Gini', 'Confidence']          
                        resultVstd = []
                        for vi, v in enumerate(resultVs):
                            biggestScore = 0
                            bestMode = modes[0]
                            for mode in modes:
                                if np.mean(modesResults[mode][vi]) > biggestScore:
                                    biggestScore = np.mean(modesResults[mode][vi])
                                    bestMode = mode 
                            resultVs[vi] = np.mean(modesResults[0][vi])
                            resultVstd.append(np.std(modesResults[0][vi]))
                            lables[vi] = lables[vi]

                        print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))
                        
                        
                        counts = np.round(resultVs,4)
                        e =  np.round(resultVstd,4)

                        ind = np.arange(len(counts))
                        rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                        rects.append(rect)
                        ax.set_xticks(ind)
                        ax.set_xticklabels(lables)
                        ax.tick_params(labelrotation=90)


    specificFolder = folderGeneral + 'GCRPerformance/'
    if not os.path.exists(specificFolder):
        os.makedirs(specificFolder)
    legend =lnames
    fig.legend(rects, labels=legend, 
                    loc="upper right", bbox_to_anchor=(1.088, 0.95)) 
    fig.tight_layout()

    fig.savefig(specificFolder + kNames[ki]+ '-GCRPerformance.png', dpi = 300, bbox_inches = 'tight')
    plt.show() 
    plt.close()

##
metricNames = ['GCR Fidelity', 'GCR Gini Index']
fig, axs = plt.subplots(nrows=len(metricNames), ncols=(len(fcamOptions) + len(gtmOptions)) , sharex=True, sharey=True,
                                figsize=(20, 14), layout="constrained")
colorN = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple']
plt.rcParams.update({'font.size': 18})



odf = pd.read_csv('baselineFullFidelity.csv')
odf = odf[['classifier_name', 'dataset_name', 'accuracy']]

odfIndex = 3144


pdf = pd.DataFrame([], columns=['classifier_name', 'dataset_name', 'accuracy'])
accs = pd.DataFrame([], columns=['method', 'Test Accuracy'])
inputIndex = 0
avgAccsIndex = 0

betterThanSimple = dict()
fullCounter = dict()
for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]):
    for gi, gcrOption in enumerate(gcrOptions):
        betterThanSimple[gcrOption] = dict()
        betterThanSimple[gcrOption]['RandomForestClassifier fidelity acc'] = 0
        betterThanSimple[gcrOption]['AdaBoostClassifier fidelity acc'] = 0
        betterThanSimple[gcrOption]['RotationForest fidelity acc'] = 0

        fullCounter[gcrOption] = dict()
        fullCounter[gcrOption]['RandomForestClassifier fidelity acc'] = 0
        fullCounter[gcrOption]['AdaBoostClassifier fidelity acc'] = 0
        fullCounter[gcrOption]['RotationForest fidelity acc'] = 0


for mi, metric in enumerate(['acc', 'giniScore']):
    jCounter = -1
    for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]):
        for gi, gcrOption in enumerate(gcrOptions):
            jCounter+=1
            
        

            
            ax = axs[mi,jCounter]
            ax.set_title(gcrNames[j2] + ' ' + fcamONames[gi] + ' ' + metricNames[mi])
            width = 0
            n_bars = 1
            standardWidth = 0.9
            bar_width = 0.9 / n_bars
            barInd = 0
            rects = []
            lNames = []
            
            cCounter = -1

            resultVs = np.zeros(len(kNames) +4)
            x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
            barInd+= 1
            width += standardWidth

            lables = kNames.copy() + ['SimpleGCR','Random Forest', 'AdaBoost', 'Rotational Forest']      

            modesResults = dict()
            
            for mode in modes:
                modesResults[mode] = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
                for symbolsCount in [5,9]: 
                    for numberFeatures in numberFeaturesSet: 
                        for symbolificationStrategy in symbolificationStrategySet:
                        
                            for j, modelSelection in enumerate([models]): 
                                for ci, hypers in enumerate(modelSelection):
                                    cCounter+=1
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
                                    configName = hypers[-1]
                                    for ni, normName in enumerate(normNames):
                                        


                                        for ki, k in enumerate(ks + ['SimpleGCR', 'RandomForestClassifier fidelity acc', 'AdaBoostClassifier fidelity acc', 'RotationForest fidelity acc']):
                                            for di in range(112):                                            

                                                dataName = dataset_list[di]
                                                        
                                                for rFolder in filteredResultsFolder:

                                                    dsName = str(dataName) +  '-n:' + str("UCR")
                                                    saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                                                    if os.path.isfile(saveName + '.pkl'):
                                                        print('FOUND: ' + saveName)
                                                        if saveName in allReses[rFolder].keys():
                                                            res = allReses[rFolder][saveName]
                                                        else:
                                                            try:
                                                                results = helper.load_obj(saveName)
                                                            except Exception as e:
                                                                print(e)
                                                                continue

                                                            res = dict()
                                                            for index, vv in np.ndenumerate(results):
                                                                res = vv

                                                            allReses[rFolder][saveName] = res
                                                    else:
                                                        print('NOT FOUND: ' + saveName)
                                                        continue

                                                    
                                                    if k == 'SimpleGCR':
                                                        for fi, f in enumerate(res['simpleGCR'][gcrOption][normName][metric]):
                                                            modesResults[mode][ki].append(res['simpleGCR'][gcrOption][normName][metric][fi])
                                                        if metric == 'acc':
                                                            if fcamONames[gi] != 'sum' and odf.query("dataset_name == @dataName").size >= 1:
                                                                odf.loc[odfIndex] = [k + ' ' + gcrNames[j2] + ' ' + fcamONames[gi] ,dataName,np.mean(res['simpleGCR'][gcrOption][normName][metric])]
                                                                odfIndex +=1
                                                            
                                                            value = k + ' selective'
                                                            if pdf.query("classifier_name == @value and dataset_name == @dataName").size >= 1:
                                                                ind = pdf.query("classifier_name == @value and dataset_name == @dataName")
                                                                if ind['accuracy'].iloc[0] < np.mean(res['simpleGCR'][gcrOption][normName][metric]):
                                                                    ind['accuracy'].iloc[0] = np.mean(res['simpleGCR'][gcrOption][normName][metric])
                                                                    pdf.loc[(pdf.classifier_name == value) & (pdf.dataset_name == dataName),'accuracy'] = np.mean(res['simpleGCR'][gcrOption][normName][metric])
                                                            else:
                                                                pdf.loc[inputIndex] = [k + ' selective', dataName,np.mean(res['simpleGCR'][gcrOption][normName][metric])]
                                                                inputIndex +=1
                                                    elif k[-3:] == 'acc':
                                                        for fi, f in enumerate(res[k]):
                                                            if metric == 'acc':
                                                                modesResults[mode][ki].append(res[k][fi])

                                                                betterThanSimple[gcrOption][k] += res[k][fi] >= res['simpleGCR'][gcrOption][normName][metric][fi]

                                                                fullCounter[gcrOption][k] += 1
                                                        if metric == 'acc':
                                                            pdf.loc[inputIndex] = [k[:-13],dataName,np.mean(res[k])]
                                                            inputIndex +=1
                                                    elif k in res['gcr'].keys():
                                                        for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName][metric]):
                                                            modesResults[mode][ki].append(res['gcr'][k][mode][gcrOption][normName][metric][fi])
                                                        if metric == 'acc':
                                                            if fcamONames[gi] != 'sum' and odf.query("dataset_name == @dataName").size >= 1:
                                                                odf.loc[odfIndex] = [kNames[ki] + ' ' + gcrNames[j2] + ' ' + fcamONames[gi],dataName,np.mean(res['gcr'][k][mode][gcrOption][normName][metric])]
                                                                odfIndex +=1
                                                            
                                                            value = kNames[ki] + ' selective'
                                                            
                                                            if pdf.query("classifier_name == @value and dataset_name == @dataName").size >= 1:
                                                                ind = pdf.query("classifier_name == @value and dataset_name == @dataName")
                                                                if ind['accuracy'].iloc[0] < np.mean(res['gcr'][k][mode][gcrOption][normName][metric]):
                                                                    ind['accuracy'].iloc[0] = np.mean(res['gcr'][k][mode][gcrOption][normName][metric])
                                                                    pdf.loc[(pdf.classifier_name == value) & (pdf.dataset_name == dataName),'accuracy'] = np.mean(res['gcr'][k][mode][gcrOption][normName][metric])

                                                            else:
                                                                pdf.loc[inputIndex] = [kNames[ki] + ' selective', dataName,np.mean(res['gcr'][k][mode][gcrOption][normName][metric])]
                                                                inputIndex +=1

                resultVstd = []
                
                for vi, v in enumerate(resultVs):
                    biggestScore = 0
                    bestMode = modes[0]

                    resultVs[vi] = np.mean(modesResults[0][vi])
                    resultVstd.append(np.std(modesResults[0][vi]))
                    lables[vi] = lables[vi]

                
                print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))
                
                
                counts = np.round(resultVs,4)
                e =  np.round(resultVstd,4)

                ind = np.arange(len(counts))
                rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                rects.append(rect)
                ax.set_xticks(ind)
                ax.set_xticklabels(lables)
                ax.set_ylim(bottom=0, top=1)
                ax.tick_params(labelrotation=90)

                if jCounter == 0:
                    ax.set_ylabel('GCR Fidelity')

specificFolder = folderGeneral + 'GCRUniformPerformance/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)



fig.tight_layout()

fig.savefig(specificFolder + 'GCRPerformance.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()

pdfM = pdf[['classifier_name', 'accuracy']].groupby(['classifier_name'])
print(pdfM)
for cn, a in pdfM:
    accs.loc[avgAccsIndex] = [cn[0], np.mean(a['accuracy'].to_numpy())]
    avgAccsIndex+= 1
pdf.to_csv(folderGeneral+'fullFidelity.csv')
accs.to_csv(folderGeneral+'avgFidelity.csv')

pdf = pdf.groupby(['classifier_name', 'dataset_name'])['accuracy'].mean().reset_index()
cdd.draw_cd_diagram(df_perf=pdf, accs=accs, folder=specificFolder,title='Fidelity', labels=True)



pdfM = odf[['classifier_name', 'accuracy']].groupby(['classifier_name'])
print(pdfM)
accs = pd.DataFrame([], columns=['method', 'Test Accuracy'])
avgAccsIndex = 0
for cn, a in pdfM:
    accs.loc[avgAccsIndex] = [cn[0], np.mean(a['accuracy'].to_numpy())]
    avgAccsIndex+= 1


odf = odf.groupby(['classifier_name', 'dataset_name'])['accuracy'].mean().reset_index()
odf.to_csv(folderGeneral+'finalODF.csv')

dns = odf.query("classifier_name == 'Attention 1st-Order r.avg. (old)'").dataset_name

print(dns)
odf = odf.query("dataset_name.isin(@dns)")

if not os.path.exists(folderGeneral +'baselineOld/'):
    os.makedirs(folderGeneral +'baselineOld/')

cdd.draw_cd_diagram(df_perf=odf, accs=accs, folder=folderGeneral + 'baselineOld/',title='Fidelity', labels=True)




f = open(folderGeneral+ "betterThanSimple.txt", "a")
for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]):
    for gi, gcrOption in enumerate(gcrOptions):
        for c in betterThanSimple[gcrOption].keys():
            f.write(gcrOption + ' ' + c + ' ' + str(betterThanSimple[gcrOption][c]/fullCounter[gcrOption][c]) + '\n')
f.close()


fig, axs = plt.subplots(nrows=2, ncols=len(fcamOptions) + len(gtmOptions), sharex=True, sharey=True,
                                figsize=(15, 12), layout="constrained")
plt.rcParams.update({'font.size': 18})
jCounter = -1
for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]): 
    for gi, gcrOption in enumerate(gcrOptions):
        jCounter+=1
        metricNames = ['Global', 'Local']
        for mi, metric in enumerate(['globalGCRAUC', 'trainLocal']):
        
            ax = axs[mi, jCounter]
            ax.set_title(gcrNames[j2] + ' ' + fcamONames[gi] + ' ' + metricNames[mi])
            print(gcrNames[j2] + ' ' +  gcrOption + ' ' + metricNames[mi])
            width = 0
            n_bars = 2
            standardWidth = 0.8
            bar_width = 0.8 / n_bars
            barInd = 0
            rects = []
            
            for mindex in [0,4]:
                resultVs = np.zeros(len(kNames))
                x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                barInd+= 1
                width += standardWidth
                normName = normNames[-1]

                modesResults = dict()
                for mode in modes:
                    modesResults[mode] = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

                    for j, modelSelection in enumerate([models]):
                        for ci, hypers in enumerate(modelSelection):
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
                            configName = hypers[-1]


                            for ki, k in enumerate(ks):
                                for di in range(112):
                                    for symbolsCount in [5,9]:
                                        for numberFeatures in numberFeaturesSet:
                                            for symbolificationStrategy in symbolificationStrategySet:

                                                dataName = dataset_list[di]
                                                        
                                                for rFolder in filteredResultsFolder:

                                                    dsName = str(dataName) +  '-n:' + str("UCR")
                                                    saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                                                    if os.path.isfile(saveName + '.pkl'):
                                                        print('FOUND: ' + saveName)
                                                        if saveName in allReses[rFolder].keys():
                                                            res = allReses[rFolder][saveName]
                                                        else:
                                                            try:
                                                                results = helper.load_obj(saveName)
                                                            except Exception as e:
                                                                print(e)
                                                                continue

                                                            res = dict()
                                                            for index, vv in np.ndenumerate(results):
                                                                res = vv

                                                            allReses[rFolder][saveName] = res
                                                    else:
                                                        print('NOT FOUND: ' + saveName)
                                                        continue

                                                    if k in res['gcr'].keys():
                                                        for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName]['acc']):
                                                            if metric == 'globalGCRAUC':
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'Reduction'][fi][0].copy()
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][mindex][0].copy()
                                                            else:
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'R'][fi].copy()
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][mindex].copy()

                                                            if len(b) == 1:
                                                                b = b[0]
                                                            if len(a) == 1:
                                                                a = a[0]

                                                            a.append(1)
                                                            b.append(0)
                                                            a = np.array(a)
                                                            a[a<0] = 0
                                                            print(np.array(a))
                                                            print(np.array(b))

                                                            print('----')

                                                            modesResults[mode][ki].append(metrics.auc(a, b))

                lables = kNames.copy() 
                resultVstd = []
                for vi, v in enumerate(resultVs):
                    biggestScore = 0
                    bestMode = modes[0]
                    for mode in modes:
                        if np.mean(modesResults[mode][vi]) > biggestScore:
                            biggestScore = np.mean(modesResults[mode][vi])
                            bestMode = mode
                    resultVs[vi] = np.mean(modesResults[0][vi])
                    resultVstd.append(np.std(modesResults[0][vi]))
                    lables[vi] = lables[vi]

                print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))
                
                
                counts = np.round(resultVs,4)
                e =  np.round(resultVstd,4)

                ind = np.arange(len(counts))
                rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3)
                rects.append(rect)
                ax.set_xticks(ind)
                ax.set_xticklabels(lables)
                ax.tick_params(labelrotation=90)
                if jCounter == 0:
                    ax.set_ylabel('GCR AUC')

specificFolder = folderGeneral + 'GCRAucPerformance/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
legend = ['Saliency AUC', 'Matching SimpleGCR AUC']
fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.234, 0.10)) 
fig.tight_layout()

fig.savefig(specificFolder + 'GCRPerformance.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()


bestIndex = dict()

for k in ks:
    for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]): 
        for gi, gcrOption in enumerate(gcrOptions):
            bestIndex[k+gcrOption] = np.zeros(5)


fig, axs = plt.subplots(nrows=2, ncols=len(fcamOptions) + len(gtmOptions), sharex=True, sharey=True,
                                figsize=(15, 8), layout="constrained")
plt.rcParams.update({'font.size': 18})
jCounter = -1
for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]): 
    for gi, gcrOption in enumerate(gcrOptions):
        jCounter+=1
        metricNames = ['Global', 'Local']
        for mi, metric in enumerate(['globalGCRAUC', 'trainLocal']):
        
            ax = axs[mi, jCounter]
            ax.set_title(gcrNames[j2] + ' ' +  gcrOption + ' '+ metricNames[mi])
            print(gcrNames[j2] + ' ' +  gcrOption + ' '+ metricNames[mi])
            width = 0
            n_bars = 3
            standardWidth = 0.8
            bar_width = 0.8 / n_bars
            barInd = 0
            rects = []
            

            mIndexes = [0,4]
            for mindex in mIndexes:
                resultVs = []
                indexVs = []
                for ku in range(len(kNames)):
                    resultVs.append([])
                    indexVs.append([])
                x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
                barInd+= 1
                width += standardWidth
                normName = normNames[-1]

                modesResults = dict()
                modesResultsR = dict()
                for mode in modes:
                    modesResults[mode] = [[],[],[],[],[],[],[],[],[],[]]
                    modesResultsR[mode] = [[],[],[],[],[],[],[],[],[],[]]

                    for j, modelSelection in enumerate([models]):
                        for ci, hypers in enumerate(modelSelection):
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
                            configName = hypers[-1]


                            for ki, k in enumerate(ks):
                                for di in range(112):
                                    for symbolsCount in [5,9]:
                                        for numberFeatures in numberFeaturesSet:
                                            for symbolificationStrategy in symbolificationStrategySet:

                                                dataName = dataset_list[di]
                                                        
                                                for rFolder in filteredResultsFolder:

                                                    dsName = str(dataName) +  '-n:' + str("UCR")
                                                    saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                                                    if os.path.isfile(saveName + '.pkl'):
                                                        print('FOUND: ' + saveName)
                                                        if saveName in allReses[rFolder].keys():
                                                            res = allReses[rFolder][saveName]
                                                        else:
                                                            try:
                                                                results = helper.load_obj(saveName)

                                                            except Exception as e:
                                                                print(e)
                                                                continue

                                                            res = dict()
                                                            for index, vv in np.ndenumerate(results):
                                                                res = vv

                                                            allReses[rFolder][saveName] = res
                                                    else:
                                                        print('NOT FOUND: ' + saveName)
                                                        continue

                                                    if k in res['gcr'].keys():
                                                        for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName]['acc']):
                                                            if metric == 'globalGCRAUC':

                                                                if mindex == -1:
                                                                    bestT = 0
                                                                    for ci, c in enumerate(res['gcr'][k][mode][gcrOption][normName][metric][fi][0]):
                                                                        if ci != 0 and len(c) > 0:
                                                                            bestT = ci
                                                                    bestIndex[k+gcrOption][bestT] += 1 
                                                                    b = res['gcr'][k][mode][gcrOption][normName][metric][fi][0][bestT].copy()
                                                                    if len(b) == 1:
                                                                        b = b[0]

                                                                else:
                                                                    b = res['gcr'][k][mode][gcrOption][normName][metric][fi][mindex][0].copy()


                                                            else:
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][mindex].copy()

                                                            if len(b) == 1:
                                                                b = b[0]
                                                            modesResults[mode][ki].append(b)
                                                            
                                                            if metric == 'globalGCRAUC':
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'Reduction'][fi][0].copy()
                                                            else:
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'R'][fi].copy()

                                                            if len(a) == 1:
                                                                a = a[0]
                                                            modesResultsR[mode][ki].append(a)

                lables = kNames.copy() 
                for vi, v in enumerate(resultVs):

                    resultVs[vi] = np.nanmean(modesResults[0][vi], axis=0)
                    lables[vi] = lables[vi]
                    indexVs[vi] = np.nanmean(modesResultsR[0][vi], axis=0)
                    if mindex == 4:
                        addition = 'Matching SimpleGCR for '
                    else:
                        addition = ''

                    ax.plot(indexVs[vi], resultVs[vi], label = addition + lables[vi])

                print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))
                

                ax.set_ylabel('GCR AUC')

specificFolder = folderGeneral + 'GCRAucPerformance/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.44, 0.40)) 

fig.tight_layout()


fig.savefig(specificFolder + 'tGCRflow.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()



f = open(folderGeneral+ "bestThold.txt", "a")

for c in bestIndex.keys():
    f.write(c + ' ' + str(bestIndex[c]) + '\n')
f.close()


fig, axs = plt.subplots(nrows=1, ncols=len(fcamOptions) + len(gtmOptions), sharex=True, sharey=True,
                                figsize=(15, 7), layout="constrained")
jCounter = -1
plt.rcParams.update({'font.size': 18})
for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]):
    for gi, gcrOption in enumerate(gcrOptions):
        jCounter+=1
        
        ax = axs[ jCounter]
        ax.set_title(gcrNames[j2]  + ' ' +  fcamONames[gi] )
        width = 0
        n_bars = 3
        standardWidth = 0.8
        bar_width = 0.8 / n_bars
        barInd = 0
        rects = []
        
        for mi, metric in enumerate(['acc','globalGCRAUC', 'trainLocal']):

            resultVs = np.zeros(len(kNames) +3)
            x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
            barInd+= 1
            width += standardWidth
            normName = normNames[-1]

            modesResults = dict()
            for mode in modes:
                modesResults[mode] = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

                for j, modelSelection in enumerate([models]):
                    for ci, hypers in enumerate(modelSelection):
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
                        configName = hypers[-1]


                        for ki, k in enumerate(ks+ ['RandomForestClassifier fidelity acc', 'AdaBoostClassifier fidelity acc','RotationForest fidelity acc']):
                            for di in range(112):
                                for symbolsCount in [5,9]:
                                    for numberFeatures in numberFeaturesSet:
                                        for symbolificationStrategy in symbolificationStrategySet:

                                            dataName = dataset_list[di]
                                                    
                                            for rFolder in filteredResultsFolder:

                                                dsName = str(dataName) +  '-n:' + str("UCR")
                                                saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                                                if os.path.isfile(saveName + '.pkl'):
                                                    print('FOUND: ' + saveName)
                                                    if saveName in allReses[rFolder].keys():
                                                        res = allReses[rFolder][saveName]
                                                    else:
                                                        try:
                                                            results = helper.load_obj(saveName)

                                                        except Exception as e:
                                                            print(e)
                                                            continue

                                                        res = dict()
                                                        for index, vv in np.ndenumerate(results):
                                                            res = vv

                                                        allReses[rFolder][saveName] = res
                                                else:
                                                    print('NOT FOUND: ' + saveName)
                                                    continue

                                                if k in betterThanSimple[gcrOption].keys() and metric == 'acc':
                                                    if fullCounter != 0:
                                                        modesResults[mode][ki].append(betterThanSimple[gcrOption][k]/fullCounter[gcrOption][k])

                                                if k in res['gcr'].keys():
                                                    if metric == 'acc':

                                                        if k in res['gcr'].keys():
                                                            for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName][metric]):
                                                                modesResults[mode][ki].append(res['gcr'][k][mode][gcrOption][normName][metric][fi] >= res['simpleGCR'][gcrOption][normName][metric][fi])


                                                    else:
                                                        for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName]['acc']):
                                                            if metric == 'globalGCRAUC':
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'Reduction'][fi][0].copy()
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][0][0].copy()
                                                            else:
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'R'][fi].copy()
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][0].copy()
                                                            a.append(1)
                                                            a = np.array(a)
                                                            a[a < 0] = 0
                                                            b.append(0)
                                                            auc1 = metrics.auc(a, b)

                                                            if metric == 'globalGCRAUC':
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'Reduction'][fi][0].copy()
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][4][0].copy()
                                                            else:
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'R'][fi].copy()
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][4].copy()
                                                            a.append(1)
                                                            a = np.array(a)
                                                            a[a < 0] = 0
                                                            b.append(0)
                                                            simpleAUC = metrics.auc(a, b)

                                                            modesResults[mode][ki].append(auc1 >= simpleAUC)

            lables = kNames.copy()+['Random Forest', 'AdaBoost', 'Rotational Forest']
            resultVstd = []
            for vi, v in enumerate(resultVs):
                biggestScore = 0
                bestMode = modes[0]
                for mode in modes:
                    if np.mean(modesResults[mode][vi]) > biggestScore:
                        biggestScore = np.mean(modesResults[mode][vi])
                        bestMode = mode
                resultVs[vi] = np.mean(modesResults[0][vi])
                resultVstd.append(np.std(modesResults[0][vi]))
                lables[vi] = lables[vi]

            print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))
            
            
            counts = np.round(resultVs,4)
            e =  np.round(resultVstd,4)

            ind = np.arange(len(counts))
            rect = ax.bar(ind+x_offset , counts, bar_width, linestyle='None', capsize=3, color=colorN[mi])
            rects.append(rect)
            ax.set_xticks(ind)
            ax.set_xticklabels(lables)
            ax.tick_params(labelrotation=90)

            ax.set_ylabel('Better Than Simple')

specificFolder = folderGeneral + 'GCRBetterThanSimpleSmall/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
legend =['Model Fidelity','Global GCR-T AUC', 'Local GCR-T AUC']
fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.118, 0.10)) 
fig.tight_layout()

fig.savefig(specificFolder + 'GCRPerformance.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()

fig, axs = plt.subplots(nrows=1, ncols=len(fcamOptions) + len(gtmOptions), sharex=True, sharey=True,
                                figsize=(15, 10), layout="constrained")
jCounter = -1
plt.rcParams.update({'font.size': 18})
for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]):
    for gi, gcrOption in enumerate(gcrOptions):
        jCounter+=1
        
        ax = axs[ jCounter]
        ax.set_title(gcrNames[j2] +  fcamONames[gi])
        width = 0
        n_bars = 3
        standardWidth = 0.8
        bar_width = 0.8 / n_bars
        barInd = 0
        rects = []
        
        for mi, metric in enumerate(['acc','globalGCRAUC', 'trainLocal']):
            if metric == 'globalGCRAUC' and j2 == 1:
                continue
            resultVs = np.zeros(len(kNames)+3)
            x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
            barInd+= 1
            width += standardWidth
            normName = normNames[-1]

            modesResults = dict()
            simpleResults = dict()
            for mode in modes:
                modesResults[mode] = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
                simpleResults[mode] = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

                for j, modelSelection in enumerate([models]):
                    for ci, hypers in enumerate(modelSelection):
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
                        configName = hypers[-1]

                        for ki, k in enumerate(ks + ['RandomForestClassifier fidelity acc', 'AdaBoostClassifier fidelity acc','RotationForest fidelity acc']):
                            if k == 'Random-Random':
                                continue
                            for di in range(112):
                                for symbolsCount in [5,9]:
                                    for numberFeatures in numberFeaturesSet:
                                        for symbolificationStrategy in symbolificationStrategySet:

                                            dataName = dataset_list[di]
                                                    
                                            for rFolder in filteredResultsFolder:

                                                dsName = str(dataName) +  '-n:' + str("UCR")
                                                saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                                                if os.path.isfile(saveName + '.pkl'):
                                                    print('FOUND: ' + saveName)
                                                    if saveName in allReses[rFolder].keys():
                                                        res = allReses[rFolder][saveName]
                                                    else:
                                                        try:
                                                            results = helper.load_obj(saveName)
                                                        except Exception as e:
                                                            print(e)
                                                            continue

                                                        res = dict()
                                                        for index, vv in np.ndenumerate(results):
                                                            res = vv

                                                        allReses[rFolder][saveName] = res
                                                else:
                                                    print('NOT FOUND: ' + saveName)
                                                    continue

                                                if k[-3:] == 'acc' and metric == 'acc':
                                                    for fi, f in enumerate(res[k]):
                                                        modesResults[mode][ki].append(res[k][fi] >= res['gcr']['Random-Random'][mode][gcrOption][normName][metric][fi])
                                                elif k in res['gcr'].keys():

                                                    if metric == 'acc':

                                                        if k in res['gcr'].keys():
                                                            for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName][metric]):
                                                                modesResults[mode][ki].append(res['gcr'][k][mode][gcrOption][normName][metric][fi] >= res['gcr']['Random-Random'][mode][gcrOption][normName][metric][fi])
                                                    else:
                                                        for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName]['acc']):
                                                            if metric == 'globalGCRAUC':
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'Reduction'][fi][0].copy()
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][0][0].copy()
                                                            else:
                                                                a = res['gcr'][k][mode][gcrOption][normName][metric +'R'][fi].copy()
                                                                b = res['gcr'][k][mode][gcrOption][normName][metric][fi][0].copy()
                                                            a.append(1)
                                                            a = np.array(a)
                                                            a[a < 0] = 0
                                                            b.append(0)
                                                            auc1 = metrics.auc(a, b)

                                                            if metric == 'globalGCRAUC':
                                                                a = res['gcr']['Random-Random'][mode][gcrOption][normName][metric +'Reduction'][fi][0].copy()
                                                                b = res['gcr']['Random-Random'][mode][gcrOption][normName][metric][fi][4][0].copy()
                                                            else:
                                                                a = res['gcr']['Random-Random'][mode][gcrOption][normName][metric +'R'][fi].copy()
                                                                b = res['gcr']['Random-Random'][mode][gcrOption][normName][metric][fi][4].copy()
                                                            a.append(1)
                                                            a = np.array(a)
                                                            a[a < 0] = 0
                                                            b.append(0)
                                                            simpleAUC = metrics.auc(a, b)

                                                            modesResults[mode][ki].append(auc1 >= simpleAUC)

            lables = kNames.copy() +['Random Forest', 'AdaBoost', 'Rotational Forest']                   
            resultVstd = []
            for vi, v in enumerate(resultVs):
                biggestScore = 0
                bestMode = modes[0]
                for mode in modes:
                    if np.mean(modesResults[mode][vi]) > biggestScore:
                        biggestScore = np.mean(modesResults[mode][vi])
                        bestMode = mode
                resultVs[vi] = np.mean(modesResults[0][vi])
                resultVstd.append(np.std(modesResults[0][vi]))
                lables[vi] = lables[vi]

            print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))
            
            
            counts = np.round(resultVs,4)
            e =  np.round(resultVstd,4)

            ind = np.arange(len(counts))
            rect = ax.bar(ind+x_offset , counts, bar_width, linestyle='None', capsize=3, color=colorN[mi])
            rects.append(rect)
            ax.set_xticks(ind)
            ax.set_xticklabels(lables)
            ax.tick_params(labelrotation=90)

            ax.set_ylabel('Percent')

specificFolder = folderGeneral + 'GCRBetterThanRandomSmall/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
legend =['GCR Fidelity','globalGCRAUC AUC', 'TrainLocal AUC']
fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.068, 0.15)) 
fig.tight_layout()

fig.savefig(specificFolder + 'GCRPerformance.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()




fig, axs = plt.subplots(nrows=len(fcamOptions) + len(gtmOptions), ncols=1, sharex=True, sharey=True,
                                figsize=(14, 15), layout="constrained")
colorN = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple']
plt.rcParams.update({'font.size': 18})
jCounter = -1
for j2, gcrOptions in enumerate([fcamOptions, gtmOptions]):
    for gi, gcrOption in enumerate(gcrOptions):
        jCounter+=1
        
        ax = axs[ jCounter]
        ax.set_title(gcrNames[j2] + ' ' +  gcrOption + ' Avg. Pos. and Neg. Diff.' )
        width = 0
        n_bars = 3
        standardWidth = 0.8
        bar_width = 0.8 / n_bars
        barInd = 0
        rects = []
        
        colorID = 0
        for mi, metric in enumerate(['acc','globalGCRAUC', 'trainLocal']):
            if metric == 'globalGCRAUC' and j2 == 1:
                continue
            resultVs = np.zeros(len(kNames))
            x_offset = (barInd - n_bars / 2) * bar_width + bar_width / 2
            barInd+= 1
            width += standardWidth
            normName = normNames[-1]

            for pos in [True, False]:
                modesResults = dict()
                
                for mode in modes:
                    modesResults[mode] = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

                    for j, modelSelection in enumerate([models]):
                        for ci, hypers in enumerate(modelSelection):
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
                            configName = hypers[-1]


                            for ki, k in enumerate(ks):
                                for di in range(112):
                                    for symbolsCount in [5,9]:
                                        for numberFeatures in numberFeaturesSet:
                                            for symbolificationStrategy in symbolificationStrategySet:

                                                dataName = dataset_list[di]
                                                        
                                                for rFolder in filteredResultsFolder:

                                                    dsName = str(dataName) +  '-n:' + str("UCR")
                                                    saveName = pt.getWeightName(dsName, dataName, 50, epochs, numOfLayers, header, dmodel, dfff, symbolsCount, 'MCB', symbolificationStrategy, numberFeatures, True, True, learning = False, results = True, resultsPath=rFolder)

                                                    if os.path.isfile(saveName + '.pkl'):
                                                        print('FOUND: ' + saveName)
                                                        if saveName in allReses[rFolder].keys():
                                                            res = allReses[rFolder][saveName]
                                                        else:
                                                            try:
                                                                results = helper.load_obj(saveName)

                                                            except Exception as e:
                                                                print(e)
                                                                continue

                                                            res = dict()
                                                            for index, vv in np.ndenumerate(results):
                                                                res = vv

                                                            allReses[rFolder][saveName] = res
                                                    else:
                                                        print('NOT FOUND: ' + saveName)
                                                        continue

                                                    if k in res['gcr'].keys():
                                                        if metric == 'acc':

                                                            if k in res['gcr'].keys():
                                                                for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName][metric]):
                                                                    diff = res['gcr'][k][mode][gcrOption][normName][metric][fi] - res['simpleGCR'][gcrOption][normName][metric][fi]
                                                                    if pos and diff > 0:
                                                                        modesResults[mode][ki].append(diff)
                                                                        f = open(folderGeneral+ "GoodPerformingDatasets.txt", "a")
                                                                        f.write(str(metric) + ' ' + str(normName) + ' ' + str(fcamONames[gi]) + ' ' +  str(kNames[ki]) + ' mode:' + str(mode) + ' fold: ' + str(fi))
                                                                        f.write("\n")
                                                                        f.write(dataName)
                                                                        f.write("\n")
                                                                        f.write('--------------------')
                                                                        f.write("\n")
                                                                        f.close()
                                                                    elif not pos and diff < 0:
                                                                        modesResults[mode][ki].append(diff)


                                                        else:
                                                            for fi, f in enumerate(res['gcr'][k][mode][gcrOption][normName]['acc']):
                                                                if metric == 'globalGCRAUC':
                                                                    a = res['gcr'][k][mode][gcrOption][normName][metric +'Reduction'][fi][0].copy()
                                                                    b = res['gcr'][k][mode][gcrOption][normName][metric][fi][0][0].copy()
                                                                else:
                                                                    a = res['gcr'][k][mode][gcrOption][normName][metric +'R'][fi].copy()
                                                                    b = res['gcr'][k][mode][gcrOption][normName][metric][fi][0].copy()
                                                                a.append(1)
                                                                a = np.array(a)
                                                                a[a < 0] = 0
                                                                b.append(0)
                                                                auc1 = metrics.auc(a, b)

                                                                if metric == 'globalGCRAUC':
                                                                    a = res['gcr'][k][mode][gcrOption][normName][metric +'Reduction'][fi][0].copy()
                                                                    b = res['gcr'][k][mode][gcrOption][normName][metric][fi][4][0].copy()
                                                                else:
                                                                    a = res['gcr'][k][mode][gcrOption][normName][metric +'R'][fi].copy()
                                                                    b = res['gcr'][k][mode][gcrOption][normName][metric][fi][4].copy()
                                                                a.append(1)
                                                                a = np.array(a)
                                                                a[a < 0] = 0
                                                                b.append(0)
                                                                simpleAUC = metrics.auc(a, b)

                                                                diff = auc1 - simpleAUC
                                                                if pos and diff > 0:
                                                                    modesResults[mode][ki].append(diff)
                                                                elif not pos and diff < 0:
                                                                    modesResults[mode][ki].append(diff)

                lables = kNames.copy() 
                resultVstd = []
                for vi, v in enumerate(resultVs):
                    bestMode =0
                    resultVs[vi] = np.mean(modesResults[0][vi])
                    resultVstd.append(np.std(modesResults[0][vi]))
                    lables[vi] = lables[vi]

                print(str(np.round(resultVs,4)) + ' +- ' + str(np.round(resultVstd,4)))
                
                
                counts = np.round(resultVs,4)
                e =  np.round(resultVstd,4)

                ind = np.arange(len(counts))
                rect = ax.bar(ind+x_offset , counts, bar_width, yerr=e, linestyle='None', capsize=3, color=list(colorN)[colorID])
                colorID += 1
                rects.append(rect)
                ax.set_xticks(ind)
                ax.set_xticklabels(lables)
                ax.tick_params(labelrotation=90)

                ax.set_ylabel('Percent')

specificFolder = folderGeneral + 'GCRBetterPerfDiff/'
if not os.path.exists(specificFolder):
    os.makedirs(specificFolder)
legend = ['Fidelity Pos. Avg.','Fidelity Neg. Avg.','Global AUC Pos. Avg.','Global AUC Neg. Avg.', 'Local AUC Pos. Avg.', 'Local AUC Neg. Avg.']
fig.legend(rects, labels=legend, 
                loc="upper right", bbox_to_anchor=(1.098, 0.08)) 
fig.tight_layout()

fig.savefig(specificFolder + 'GCRPerformance.png', dpi = 300, bbox_inches = 'tight')
plt.show() 
plt.close()

# %%
