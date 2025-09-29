from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients, TokenReferenceBase, visualization,Saliency,DeepLift,DeepLiftShap,GradientShap,InputXGradient,GuidedBackprop,LRP,GuidedGradCam,FeatureAblation,KernelShap,Deconvolution,FeaturePermutation,LimeBase,LRP,ShapleyValueSampling
import torch
import numpy as np
from modules import helper
import shap
from modules import pytorchTrain as pt
from modules import GCRPlus
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn import metrics
from modules.games import games
from modules.approximators import SHAPIQEstimator
from modules import gini

import psutil


ram = psutil.virtual_memory()

def printCheck(path, msg):
    f = open(path+ "notes.txt", "a")
    f.write(msg +'\n')
    f.write("RAM usage (%):"+ str(ram.percent) + '\n')
    f.write("RAM used (GB):"+ str(round(ram.used / 1e9, 2)) + '\n')
    f.close()

def getRelpropSaliency(device, data, model, method=None, outputO = None, batchSize=5000):
        outRel = None
        for batch_start in range(0, len(data), batchSize):
            batchEnd = batch_start + batchSize      
            input_ids = torch.from_numpy(data[batch_start:batchEnd]).to(device) 
            input_ids.requires_grad_()
            output = model(input_ids)

            if outputO is None:
                outputOut = output.cpu().data.numpy()#[0]
                index = np.argmax(outputOut, axis=-1)
                one_hot = np.zeros((outputOut.shape[0], outputOut.shape[-1]), dtype=np.float32)
                for h in range(len(one_hot)):
                    one_hot[h, index[h]] = 1
            else:
                outputOut = outputO[batch_start:batchEnd]
                index = np.argmax(outputOut, axis=-1)
                one_hot = outputOut

            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            model.zero_grad()
            one_hot.backward(retain_graph=True)
            one_hot.shape
            kwargs = {"alpha": 1}


            if method:
                outRelB = model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), method=method, **kwargs).cpu().detach().numpy()
            else:
                outRelB = model.relprop(torch.tensor(one_hot_vector).to(input_ids.device) , **kwargs).cpu().detach().numpy()
            
            if outRel is None:
                outRel = outRelB
            else:
                outRel = np.vstack([outRel, outRelB])
        return outRel

def reshape_transform(tensor):
    result = tensor.reshape(tensor.size(0), tensor.size(1), 1, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def interpret_dataset(sal, data, targets, package='PytGradCam', smooth=False):
    if package == 'captum':
        attributions_ig = sal.attribute(data, target=targets)
        return attributions_ig.cpu().squeeze().detach().numpy()
    elif package == 'PytGradCam':

        grayscale_cam = sal(input_tensor=data, targets=targets, eigen_smooth=smooth)

        grayscale_cam = grayscale_cam
        return grayscale_cam

def fcamQualityMetric(rGM):
    giniScore = gtmQualityMetric(rGM)


    return giniScore


def fcamAUCMetric(rGM, rM, valuesA, testdata, ranges, combis, predictions, percentSteps=20, globalT=True,calcCompareSimpleGCR=False, rMA=True):
    print('Do auc score')
    steps = np.array(list(range(0, 100, percentSteps))) /100
    minScores = GCRPlus.calcFCAMMinScoreNP(rGM)
    gcrSFcamOutA = GCRPlus.classFullAttFast(rGM, testdata, ranges, combis, predictions, rM, minScores, valuesA, useThreshold=True, thresholdPercents=steps, calcCompareSimpleGCR=calcCompareSimpleGCR, rMA=rMA)

    return gcrSFcamOutA

def gtmQualityMetric(rGM):
    giniScore = 0


    giniScore = gini.gini(rGM)


    return giniScore

def gtmAUCMetric(rGM, gtmAbstraction, ix_test, minScoresGTM, predictions, dataLen, rM, percentSteps=20, reductionName='MixedClasses', calcCompareSimpleGCR=False):
    print('Do auc score')
    steps = np.array(list(range(0, 100, percentSteps))) /100
    gcrSGTMOutA = GCRPlus.calcFullAbstractAttentionFast(rGM, gtmAbstraction, ix_test, minScoresGTM, dataLen, predictions, rM, useThreshold=True, thresholdPercents=steps,calcCompareSimpleGCR=calcCompareSimpleGCR)

    return gcrSGTMOutA

def splitSaliency(num_of_classes, outRel, targets, do3DData=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='max'):

    if do3DData:
        outRel = helper.doCombiStep(op1, outRel, axis1)
        outRel = helper.doCombiStep(op2, outRel, axis2) 
        outRel = helper.doCombiStep(op3, outRel, axis3) 
    avgOut = []
    argsOut = []
    for c in range(num_of_classes):
        argsVC = np.argwhere(np.argmax(targets, axis=1)==c).flatten()
        if len(outRel[argsVC]) == 1:
            meanC = np.mean(outRel[argsVC].squeeze(), axis=0) 
        else:
            meanC = outRel[argsVC].squeeze()
        avgOut.append(meanC)
        argsOut.append(argsVC)

    return avgOut, argsOut

def mapSaliency(output, num_of_classes, outRel, y_train, outVal, y_val, outTest, y_test, do3DData=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='max'):
    cTrain, argTrain = splitSaliency(num_of_classes, outRel, y_train, do3DData=do3DData, axis1=axis1, axis2=axis2, axis3=axis3, op1=op1, op2=op2, op3=op3)
    caseVal, argVal= splitSaliency(num_of_classes, outVal, y_val, do3DData=do3DData, axis1=axis1, axis2=axis2, axis3=axis3, op1=op1, op2=op2, op3=op3)
    cTest, argTest= splitSaliency(num_of_classes, outTest, y_test, do3DData=do3DData, axis1=axis1, axis2=axis2, axis3=axis3, op1=op1, op2=op2, op3=op3)
    if  'ClassTrain' not in output:
        output['ClassTrain']= []
        output['ArgClassTrain']= []
        output['ClassVal']= []
        output['ArgClassVal']= []  
        output['ClassTest']= []
        output['ArgClassTest']= []
    output['ClassTrain'].append(cTrain)
    output['ArgClassTrain'].append(argTrain) 
    output['ClassVal'].append(caseVal)
    output['ArgClassVal'].append(argVal)     
    output['ClassTest'].append(cTest)
    output['ArgClassTest'].append(argTest)

def splitPerValue(valueSaliency, saliency, x_train1, nrSymbols):
    inputIds = x_train1.squeeze()
    sValues = saliency.squeeze()
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)

    for a in symbolA:
        if a not in valueSaliency.keys():
            valueSaliency[a] = []
            for l in range(len(inputIds[0])):
                valueSaliency[a].append([])

    for n in range(len(inputIds)): 
        for k in range(len(inputIds[n])):
            valueSaliency[round(float(inputIds[n][k]), 4)][k].append(sValues[n][k]) 

    return valueSaliency

def splitPerValueAndClass(valueSaliency, saliency, x_train1, nrSymbols, numerOfClasses, lables):
    targets = np.argmax(lables, axis=1)
    inputIds = x_train1.squeeze()
    sValues = saliency.squeeze()
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)

    for c in range(numerOfClasses):
        if c not in valueSaliency.keys():
            valueSaliency[c] = dict()
        for a in symbolA:
            if a not in valueSaliency[c].keys():
                valueSaliency[c][a] = []
                for l in range(len(inputIds[0])):
                    valueSaliency[c][a].append([])

    for n in range(len(inputIds)): 
        for k in range(len(inputIds[n])):
            valueSaliency[targets[n]][round(float(inputIds[n][k]), 4)][k].append(sValues[n][k]) 

    return valueSaliency
        
def reduceMap(saliencyMap, do3DData=False, do3D2Step=False, do3rdStep=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='sum'):#NOTE op3 is sum
    saliencyMapReturn = saliencyMap.copy()
    if do3DData:
        saliencyMapReturn = helper.doCombiStep(op1, saliencyMapReturn, axis1)
        saliencyMapReturn = helper.doCombiStep(op2, saliencyMapReturn, axis2) 
        saliencyMapReturn = helper.doCombiStep(op3, saliencyMapReturn, axis3) 
    elif do3D2Step:
        saliencyMapReturn = helper.doCombiStep(op1, saliencyMapReturn, axis1)
        saliencyMapReturn = helper.doCombiStep(op2, saliencyMapReturn, axis2) 
    elif do3rdStep:
        saliencyMapReturn = helper.doCombiStep(op3, saliencyMapReturn, axis3) 

    return saliencyMapReturn

#gives how often inputs with information is below the irrelevant data
def getPredictionSaliency(nrEmpty, saliencys, x_train1, classes, y_train1, nrSymbols, andStack, orStack, xorStack, nrAnds, nrOrs, nrxor, orOffSet, xorOffSet, trueIndexes):
    targets = np.argmax(y_train1, axis=1)
    
    inputIds = x_train1.squeeze()            
    symbolA = helper.getMapValues(nrSymbols)
    symbolA = np.array(symbolA)
    trueSymbols = symbolA[trueIndexes]       

    wrongImportanceMeaning = dict() 
    countImportanceMeaning = dict()
    wrongGIB = 0
    for c in classes:
        wrongImportanceMeaning[int(c)] =  np.zeros(len(x_train1[0]))
        countImportanceMeaning[int(c)] =  np.zeros(len(x_train1[0]))
       
            
    for n in range(len(inputIds)):
        baseline = np.max(saliencys[n][-1* nrEmpty:])
        if targets[n] == 0:
            for j in range(andStack):
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
        if targets[n] == 1:
            for j in range(andStack):
                for k in range((nrAnds*j), nrAnds*(j+1)):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1

        maxAnds = (nrAnds * andStack)


        if targets[n] == 1:
            for j in range(orStack):
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
                        
        if targets[n] == 0:
            for j in range(orStack):
                for k in range(maxAnds + (nrOrs *j - orOffSet * j),maxAnds+(nrOrs *(j+1) - orOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1

        maxOrs = nrOrs * orStack - orOffSet * orStack


        for j in range(xorStack):
            all0 = True
            highest = 0
            ndHighest = 0
            k1 = -1
            k2 = -1
            if targets[n] == 0:
                for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                    if round(float(inputIds[n][k]), 4) not in trueSymbols:
                        all0 = False
                        break
                    if saliencys[n][k] > highest:
                        ndHighest = highest
                        highest=saliencys[n][k]
                        k2 = k1
                        k1 = k
                    elif saliencys[n][k] > ndHighest:
                        ndHighest = saliencys[n][k]
                        k2 = k

                if not all0: 
                    if saliencys[n][k1] < baseline:
                        wrongImportanceMeaning[targets[n]][k1] += 1
                    countImportanceMeaning[targets[n]][k1] += 1  
                    if saliencys[n][k2] < baseline:
                        wrongImportanceMeaning[targets[n]][k2] += 1
                    countImportanceMeaning[targets[n]][k2] += 1       

            for k in range(maxAnds+maxOrs+(nrxor *j - xorOffSet * j),maxAnds+maxOrs+(nrxor *(j+1) - xorOffSet * (j+1))):
                if targets[n] == 1:
                    if saliencys[n][k] < baseline:
                        wrongImportanceMeaning[targets[n]][k] += 1
                    countImportanceMeaning[targets[n]][k] += 1       
                elif targets[n] == 0: 
                    if all0:
                        if saliencys[n][k] < baseline:
                            wrongImportanceMeaning[targets[n]][k] += 1
                        countImportanceMeaning[targets[n]][k] += 1
                    else:
                        break      

    return wrongImportanceMeaning, countImportanceMeaning
        

def getAttention(device, model, x_train1, y_train1, val_batch_size=50, batch_size=50, doClsTocken=False):
        fullAttention = pt.validate_model(device, model, x_train1, y_train1, val_batch_size, batch_size, '', output_attentions=True)
        fullAttention = np.concatenate(fullAttention)
     
        return fullAttention

def buildSHAP2D(approx_value, n_sii_order, shapley_extractor_sii):
    n_shapley_values = shapley_extractor_sii.transform_interactions_in_n_shapley(interaction_values=approx_value, n=n_sii_order, reduce_one_dimension=False)
    for i, v in enumerate(n_shapley_values[1]):
        n_shapley_values[2][i][i] = v
    return n_shapley_values[2]

def getIQSHAP(data, model, modelType):
    metaGame = games.DLMetaGame(model, data, modelType)

    interaction_order = 2
    budget = 2**7
    outShap = []
    for l in range(len(data)):
        game = games.DLGame(
            meta_game= metaGame,
            data_index=l
        )
        game_name = game.game_name
        game_fun = game.set_call
        n = game.n
        N = set(range(n))
        shapley_extractor_sii = SHAPIQEstimator(
            N=N,
            order=interaction_order,
            interaction_type="SII",
            top_order=False
        )
        approx_value = shapley_extractor_sii.compute_interactions_from_budget(
            game=game.set_call,
            budget=budget,
            pairing=False
        )
        shapResults = buildSHAP2D(approx_value, interaction_order, shapley_extractor_sii)
        outShap.append(shapResults)
    return outShap

def getSaliencyMap(outMap, saveKey, device, numberOfLables, modelType: str, method: str, submethod: str, model, x_train, x_val, x_test, y_train, y_val, y_test, smooth=False, batches=True, batchSize=50, doClassBased=True, doClsTocken=False, fullSave=False):
    outTrain = []
    outVal = []
    outTest = []
    print('methods:')
    print(method)
    print(submethod)

    if not batches:
        batchSize = len(y_train)
    
    if method == 'LRP':
            outTrain = getRelpropSaliency(device, x_train, model, method=submethod, batchSize=batchSize)
            outVal = getRelpropSaliency(device, x_val, model, method=submethod, batchSize=batchSize)
            outTest = getRelpropSaliency(device, x_test, model, method=submethod, batchSize=batchSize)
            if fullSave:
                for lable in range(numberOfLables):
                    targets = np.zeros((x_train.shape[0], numberOfLables), dtype=np.float32)
                    for t in range(len(targets)):
                        targets[t, lable] = 1
                    outTrainC = getRelpropSaliency(device, x_train, model, method=submethod, outputO=targets, batchSize=batchSize)
                    targets = np.zeros((x_val.shape[0], numberOfLables), dtype=np.float32)
                    for t in range(len(targets)):
                        targets[t, lable] = 1
                    outValC = getRelpropSaliency(device, x_val, model, method=submethod, outputO=targets, batchSize=batchSize)
                    targets = np.zeros((x_test.shape[0], numberOfLables), dtype=np.float32)
                    for t in range(len(targets)):
                        targets[t, lable] = 1
                    outTestC = getRelpropSaliency(device, x_test, model, method=submethod, outputO=targets, batchSize=batchSize)
                    
                    outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainC)
                    outMap['classes'][str(lable)][saveKey + 'Val'].append(outValC)
                    outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestC)
    elif method == 'Random':
        x_trainf = x_train.squeeze()
        x_valf = x_val.squeeze()
        x_testf = x_test.squeeze()
        outTrain = np.random.randn(x_trainf.shape[0], x_trainf.shape[1], x_trainf.shape[1])
        outVal = np.random.randn(x_valf.shape[0], x_valf.shape[1], x_valf.shape[1])
        outTest = np.random.randn(x_testf.shape[0], x_testf.shape[1], x_testf.shape[1])
    elif method == 'IQ-SHAP':
        if submethod == "2OrderShap":
            outTrain = getIQSHAP(x_train, model,modelType)
            outVal = getIQSHAP(x_val, model,modelType)
            outTest = getIQSHAP(x_test, model,modelType)
    elif method ==  'captum':
            if submethod == "IntegratedGradients":
                    lig = IntegratedGradients(model)
                    batchSize = 1
            elif submethod == "Saliency":
                    lig = Saliency(model)
            elif submethod == "DeepLift":
                    lig = DeepLift(model)
            elif submethod == "KernelShap":
                    lig = KernelShap(model)
            elif submethod == "InputXGradient":
                    lig = InputXGradient(model)
            elif submethod == "GuidedBackprop":
                    lig = GuidedBackprop(model)
            elif submethod == "GuidedGradCam":
                    if modelType == "Transformer":
                        lig = GuidedGradCam(model, layer=model.encoder.layer[-1])
                    elif modelType == "CNN":
                        lig = GuidedGradCam(model, layer=model.lastConv)
                    else:
                        raise ValueError("Not a valid model type for gradcam")
            elif submethod == "FeatureAblation":
                    lig = FeatureAblation(model)
            elif submethod == "FeaturePermutation":
                    lig = FeaturePermutation(model)
            elif submethod == "Deconvolution":
                    lig = Deconvolution(model)
            else:
                    raise ValueError("Not a valid captum submethod")

            if doClassBased:
                maxGoal = numberOfLables
            else:
                maxGoal = 0
            for lable in range(-1, maxGoal):
                outTrainA = None
                outValA = None
                outTestA = None

                oldBatchSize = batchSize
                counter = 0
                while (len(y_train) % batchSize) == 1:
                    batchSize += 1
                    counter +=1
                    if counter == 10:
                        break

                for batch_start in range(0, len(y_train), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_train[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_train[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_train[batch_start:batchEnd]).to(device) 

                    outTrainB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTrainA is None:
                        outTrainA = outTrainB
                    else:
                        outTrainA = np.vstack([outTrainA,outTrainB])

                
                counter = 0
                batchSize = oldBatchSize
                while (len(y_val) % batchSize) == 1:
                    batchSize += 1
                    counter +=1
                    if counter == 10:
                        break
                
                for batch_start in range(0, len(y_val), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_val[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_val[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_val[batch_start:batchEnd]).to(device) 
                    outValB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outValA is None:
                        outValA = outValB
                    else:
                        outValA = np.vstack([outValA,outValB])

                counter = 0
                batchSize = oldBatchSize

                while (len(y_test) % batchSize) == 1:
                    batchSize += 1
                    counter +=1
                    if counter == 10:
                        break
                for batch_start in range(0, len(y_test), batchSize):
                    batchEnd = batch_start + batchSize
                    if lable == -1:           
                        targets = torch.from_numpy(y_test[batch_start:batchEnd].argmax(axis=1)).to(device) 
                    else:
                        targets = torch.from_numpy(np.zeros((len(y_test[batch_start:batchEnd])), dtype=np.int64) + lable).to(device) 
                    input_ids = torch.from_numpy(x_test[batch_start:batchEnd]).to(device) 
                    outTestb = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                    if outTestA is None:
                        outTestA = outTestb
                    else:
                        outTestA = np.vstack([outTestA, outTestb])
                if lable == -1:
                    outTrain = outTrainA
                    outVal = outValA
                    outTest = outTestA
                else:
                    outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainA)
                    outMap['classes'][str(lable)][saveKey + 'Val'].append(outValA)
                    outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestA)

    elif method ==  'PytGradCam':
            if modelType == "Transformer":
                layer=model.encoder.layer[-1]
                reshapeMethod = reshape_transform
            elif modelType == "CNN":
                layer=model.lastConv
                reshapeMethod = None
            else:
                raise ValueError("Not a valid model type for PytGradCam")

            if submethod == "EigenCAM":
                    lig = EigenCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "GradCAMPlusPlus":
                    lig = GradCAMPlusPlus(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "XGradCAM":
                    lig = XGradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "GradCAM":
                    lig = GradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)
            elif submethod == "EigenGradCAM":
                    lig = EigenGradCAM(model, target_layers=[layer], use_cuda=True, reshape_transform=reshapeMethod)

            else:
                    raise ValueError("Not a valid PytGradCam submethod")

            if not batches:
                batchSize = len(y_train)

            for lable in range(-1, numberOfLables):
                if fullSave or lable == -1:

                    outTrainA = None
                    outValA = None
                    outTestA = None
                    for batch_start in range(0, len(y_train), batchSize):
                        batchEnd = batch_start + batchSize
                        input_ids = torch.from_numpy(x_train[batch_start:batchEnd]).to(device) 

                        
                        target_categories = torch.from_numpy(y_train[batch_start:batchEnd].argmax(axis=1)).to(device) 
                        if lable != -1:           
                            target_categories = target_categories * 0 + lable

                        target_categories = target_categories.squeeze()
                        targets = [ClassifierOutputTarget(category) for category in target_categories]

                        outTrainB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                        if outTrainA is None:
                            outTrainA = outTrainB
                        else:
                            outTrainA = np.vstack([outTrainA,outTrainB])

                    for batch_start in range(0, len(y_val), batchSize):
                        batchEnd = batch_start + batchSize
                        target_categories = torch.from_numpy(y_val[batch_start:batchEnd].argmax(axis=1)).to(device) 
                        if lable != -1:           
                            target_categories = target_categories * 0 + lable

                        target_categories = target_categories.squeeze()
                        targets = [ClassifierOutputTarget(category) for category in target_categories]

                        input_ids = torch.from_numpy(x_val[batch_start:batchEnd]).to(device) 

                        
                        outValB = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                        if outValA is None:
                            outValA = outValB
                        else:
                            outValA = np.vstack([outValA,outValB])

                    for batch_start in range(0, len(y_test), batchSize):
                        batchEnd = batch_start + batchSize
                        target_categories = torch.from_numpy(y_test[batch_start:batchEnd].argmax(axis=1)).to(device) 
                        if lable != -1:           
                            target_categories = target_categories * 0 + lable

                        target_categories = target_categories.squeeze()
                        targets = [ClassifierOutputTarget(category) for category in target_categories]

                        input_ids = torch.from_numpy(x_test[batch_start:batchEnd]).to(device) 

                        
                        outTestb = interpret_dataset(lig, input_ids, targets, package=method, smooth=smooth)
                        if outTestA is None:
                            outTestA = outTestb
                        else:
                            outTestA = np.vstack([outTestA, outTestb])
                    if lable == -1:
                        outTrain = outTrainA
                        outVal = outValA
                        outTest = outTestA
                    else:
                        outMap['classes'][str(lable)][saveKey + 'Train'].append(outTrainA)
                        outMap['classes'][str(lable)][saveKey + 'Val'].append(outValA)
                        outMap['classes'][str(lable)][saveKey + 'Test'].append(outTestA)


    elif method ==  'SHAP':
            explainer = shap.TreeExplainer(model, x_train)
            outTrain = explainer.shap_values(x_train)
            
            explainer = shap.TreeExplainer(model, x_val)
            outVal = explainer.shap_values(x_val)
            
            explainer = shap.TreeExplainer(model, x_test)
            outTest = explainer.shap_values(x_test)
            
    elif method == 'Attention': 
            #axis1 = 2
            #axis2 = 0
            #axis3 = 1
            batchSize = 1

            outTrain = getAttention(device, model, x_train, y_train, val_batch_size=batchSize, batch_size=batchSize, doClsTocken=doClsTocken)
            outVal = getAttention(device,model, x_val, y_val, val_batch_size=batchSize, batch_size=batchSize, doClsTocken=doClsTocken)
            outTest = getAttention(device, model, x_test, y_test, val_batch_size=batchSize, batch_size=batchSize, doClsTocken=doClsTocken)

    else:
        print('unknown saliency method: ' + method)
        raise Exception('Unknown Saliency Method: ' + method)

    outTrain = np.array(outTrain).squeeze()
    outVal = np.array(outVal).squeeze()
    outTest = np.array(outTest).squeeze()

    outMap[saveKey + 'Train'].append(outTrain)
    outMap[saveKey + 'Val'].append(outVal)
    outMap[saveKey + 'Test'].append(outTest)
    if fullSave:
        outMap['means'][saveKey + 'Train'].append(np.mean(outTrain.squeeze(), axis=1))
        outMap['means'][saveKey + 'Val'].append(np.mean(outVal.squeeze(), axis=1))
        outMap['means'][saveKey + 'Test'].append(np.mean(outTest.squeeze(), axis=1))

    do3DData = False
    do2DData = False
    if len(outTrain.shape) > 3:
        do3DData = False
        outTrain = reduceMap(outTrain, do3D2Step=True, do3rdStep=False)
        outVal = reduceMap(outVal, do3D2Step=True, do3rdStep=False)
        outTest = reduceMap(outTest, do3D2Step=True, do3rdStep=False)
        do2DData=True
    elif len(outTrain.shape) > 2:
        do2DData = True


    return outTrain, outVal, outTest, do3DData, do2DData
    
def transformMod(sd, mode):
    if mode == 2:
        sd[sd < 0] = 0 
    elif mode == 3:
        sd = np.absolute(sd)
    return sd

def doSimpleGCR(traindata, trainLables, testdata, testLables, num_of_classes, symbolsCount, trainCombis, ranges, rangesSmall, testCombis, gtmRange, reductionName="MixedClasses", addMaskedValue=False, ignoreMaskedValue=False, addOne=False, doMetrics=False, doSoloGini=True):
    fullAttention = np.zeros(traindata.shape, dtype=float)
    fullAttention  = fullAttention+1

    return do3DGCR(fullAttention, traindata, trainLables, testdata, testLables, num_of_classes, symbolsCount, trainCombis, ranges, rangesSmall, testCombis, gtmRange, reductionName=reductionName, doMetrics=doMetrics, addMaskedValue=addMaskedValue, ignoreMaskedValue=ignoreMaskedValue, doMax=False, doPenalty=False, penaltyMode="entropy", addOne=False, order = 'lh', step1 = 'sum', step2 = 'sum', do3DData=False, do2DData=False, doSoloGini=True)

def doLasaAuc3DGCR(resultDict, fullAttention, testAttention, traindata, trainLables, testdata, testLables, num_of_classes, symbolsCount, trainCombis, ranges, rangesSmall, testCombis, gtmRange, saliencyCombis, reductionName="MixedClasses", mode=0, threshold=-1, percentSteps=20, doMax=False, doPenalty=False, penaltyMode="entropy", addOne=False, useRM = True, order = 'lh', step1 = 'sum', step2 = 'sum', do3DData=True, do2DData=False, doGTM=True):   



        fullAttentionM = transformMod(fullAttention, mode)
        saliencyCombisM = transformMod(saliencyCombis, mode)

        ignoreMaskedValue = True
        addMaskedValue = True
        calcCompareSimpleGCR = True
        doMinimalMetrics=True
        #mode = 0

        for rk in resultDict.keys():
            resultDict[rk][reductionName]['trainGlobal'].append([[],[],[],[],[],[],[],[]])
            resultDict[rk][reductionName]['trainLocal'].append([[],[],[],[],[],[],[],[]])

            resultDict[rk][reductionName]['trainGlobalR'].append([])
            resultDict[rk][reductionName]['trainLocalR'].append([])

            resultDict[rk][reductionName]['trainGlobalGini'].append([])
            resultDict[rk][reductionName]['trainLocalGini'].append([])

            resultDict[rk][reductionName]['globalGCRAUC'].append([[],[],[],[],[],[],[],[]])
            resultDict[rk][reductionName]['globalGCRAUCReduction'].append([])
            resultDict[rk][reductionName]['globalGCRAUCGini'].append([])
            resultDict[rk][reductionName]['globalGCRAUCConfidence'].append([])


        newTrainLocalData = []
        newTrainCombisLocalData = []
        for thresholdFactor in (np.array(list(range(0, 100, percentSteps))) /100):
            newTrainLocal, newTrainLocalR,newTrainCombisLocal = doSimpleLasaReduction(fullAttentionM, traindata, thresholdFactor, trainCombis, saliencyCombisM, maskValue=symbolsCount,doFidelity=False, do3DData=do3DData, do3rdStep=do2DData, globalT=False)
            newTrainLocalData.append(newTrainLocal)
            newTrainCombisLocalData.append(newTrainCombisLocal)





            doMetrics = doMinimalMetrics and thresholdFactor == 0

            trainLocal = do3DGCR(fullAttention, newTrainLocal, trainLables, testdata, testLables, num_of_classes, symbolsCount,  np.array(newTrainCombisLocal), ranges, rangesSmall, testCombis, gtmRange, mode=mode, reductionName=reductionName, doMetrics=doMetrics, threshold=threshold, addMaskedValue=addMaskedValue, ignoreMaskedValue=ignoreMaskedValue, calcCompareSimpleGCR=calcCompareSimpleGCR, doMax=doMax, doPenalty=doPenalty, penaltyMode=penaltyMode, useRM=useRM,addOne=addOne, order = order, step1 = step1, step2 = step2, do3DData=do3DData, do2DData=do2DData,doGTM=doGTM)
            

            if doMetrics:
                for vi in range(len(trainLocal[0][6][0])):
                    resultDict['rMS'][reductionName]['globalGCRAUC'][-1][vi].append(trainLocal[0][6][0][vi])
                    resultDict['rMA'][reductionName]['globalGCRAUC'][-1][vi].append(trainLocal[1][6][0][vi])
                resultDict['rMS'][reductionName]['globalGCRAUCReduction'][-1].append(trainLocal[0][6][1][2])
                resultDict['rMA'][reductionName]['globalGCRAUCReduction'][-1].append(trainLocal[1][6][1][2])
                resultDict['rMA'][reductionName]['trainLocalGini'][-1].append(trainLocal[1][5])
                resultDict['rMS'][reductionName]['trainLocalGini'][-1].append(trainLocal[0][5])
                resultDict['rMS'][reductionName]['globalGCRAUCConfidence'][-1].append(trainLocal[0][6][4])
                resultDict['rMA'][reductionName]['globalGCRAUCConfidence'][-1].append(trainLocal[1][6][4])
                resultDict['rMS'][reductionName]['globalGCRAUCGini'][-1].append(trainLocal[0][5])
                resultDict['rMA'][reductionName]['globalGCRAUCGini'][-1].append(trainLocal[1][5])
            else:
                for vi in range(8):
                    resultDict['rMS'][reductionName]['globalGCRAUC'][-1][vi].append([])
                    resultDict['rMA'][reductionName]['globalGCRAUC'][-1][vi].append([])
                resultDict['rMS'][reductionName]['globalGCRAUCGini'][-1].append([])
                resultDict['rMA'][reductionName]['globalGCRAUCGini'][-1].append([])
                resultDict['rMS'][reductionName]['globalGCRAUCReduction'][-1].append([])
                resultDict['rMA'][reductionName]['globalGCRAUCReduction'][-1].append([])
                resultDict['rMA'][reductionName]['trainLocalGini'][-1].append([])
                resultDict['rMS'][reductionName]['trainLocalGini'][-1].append([])
                resultDict['rMS'][reductionName]['globalGCRAUCConfidence'][-1].append([])
                resultDict['rMA'][reductionName]['globalGCRAUCConfidence'][-1].append([])



            resultDict['rMS'][reductionName]['trainLocalR'][-1].append(np.mean(newTrainLocalR))

            for vi, v in enumerate(trainLocal[0][0]):
                resultDict['rMS'][reductionName]['trainLocal'][-1][vi].append(v)


            resultDict['rMA'][reductionName]['trainLocalR'][-1].append(np.mean(newTrainLocalR))


            for vi, v in enumerate(trainLocal[1][0]):
                resultDict['rMA'][reductionName]['trainLocal'][-1][vi].append(v)

                

            for k in GCRPlus.gtmReductionStrings():
                resultDict[k][reductionName]['trainLocalR'][-1].append(np.mean(newTrainLocalR))

                for vi, v in enumerate(trainLocal[2][k]['performance']):
                    resultDict[k][reductionName]['trainLocal'][-1][vi].append(v)


                if doMetrics:
                    for vi in range(len(trainLocal[2][k]['aucScoreGlobal'])):
                        resultDict[k][reductionName]['globalGCRAUC'][-1][vi].append(trainLocal[2][k]['aucScoreGlobal'][vi])
                    resultDict[k][reductionName]['globalGCRAUCReduction'][-1].append(trainLocal[2][k]['aucScoreGlobalReduction'])
                    resultDict[k][reductionName]['trainLocalGini'][-1].append(trainLocal[2][k]['giniScore'])
                    resultDict[k][reductionName]['globalGCRAUCConfidence'][-1].append(trainLocal[2][k]['confidence'])
                    resultDict[k][reductionName]['globalGCRAUCGini'][-1].append(trainLocal[2][k]['giniScore'])
                else:
                    for vi in range(8):
                        resultDict[k][reductionName]['globalGCRAUC'][-1][vi].append([])
                    resultDict[k][reductionName]['globalGCRAUCReduction'][-1].append([])
                    resultDict[k][reductionName]['trainLocalGini'][-1].append([])
                    resultDict[k][reductionName]['globalGCRAUCConfidence'][-1].append([])
                    resultDict[k][reductionName]['globalGCRAUCGini'][-1].append([])


            if thresholdFactor == 0:
                resultDict['rMS'][reductionName]['acc'].append(trainLocal[0][0][0])
                resultDict['rMS'][reductionName]['predicsion'].append(trainLocal[0][0][1])
                resultDict['rMS'][reductionName]['recall'].append(trainLocal[0][0][2])
                resultDict['rMS'][reductionName]['f1'].append(trainLocal[0][0][3])
                resultDict['rMS'][reductionName]['confidence'].append(trainLocal[0][4])
                


                resultDict['rMA'][reductionName]['acc'].append(trainLocal[1][0][0])
                resultDict['rMA'][reductionName]['predicsion'].append(trainLocal[1][0][1])
                resultDict['rMA'][reductionName]['recall'].append( trainLocal[1][0][2])
                resultDict['rMA'][reductionName]['f1'].append(trainLocal[1][0][3])

                resultDict['rMA'][reductionName]['confidence'].append(trainLocal[1][4])
                if doMetrics:
                    resultDict['rMS'][reductionName]['giniScore'].append(trainLocal[0][5])
                    resultDict['rMA'][reductionName]['giniScore'].append(trainLocal[1][5])

                for gtmAbstact in GCRPlus.gtmReductionStrings():
                    resultDict[gtmAbstact][reductionName]['acc'].append(trainLocal[2][gtmAbstact]['performance'][0])
                    resultDict[gtmAbstact][reductionName]['predicsion'].append(trainLocal[2][gtmAbstact]['performance'][1])
                    resultDict[gtmAbstact][reductionName]['recall'].append(trainLocal[2][gtmAbstact]['performance'][2])
                    resultDict[gtmAbstact][reductionName]['f1'].append(trainLocal[2][gtmAbstact]['performance'][3])
                    resultDict[gtmAbstact][reductionName]['confidence'].append(trainLocal[2][gtmAbstact]['confidence'])
                    if doMetrics:
                        resultDict[gtmAbstact][reductionName]['giniScore'].append(trainLocal[2][gtmAbstact]['giniScore'])


        if doMinimalMetrics:
            bestPerformanceIndex = np.argmax(np.array(resultDict['rMA'][reductionName]['trainLocal'][-1][0]))

            if bestPerformanceIndex != 0:
                
                newTrainLocal = newTrainLocalData[bestPerformanceIndex]
                trainLocal = do3DGCR(fullAttention, newTrainLocal, trainLables, testdata, testLables, num_of_classes, symbolsCount,  np.array(newTrainCombisLocalData[bestPerformanceIndex]), ranges, rangesSmall, testCombis, gtmRange, mode=mode, reductionName=reductionName, doMetrics=True, threshold=threshold, addMaskedValue=addMaskedValue, ignoreMaskedValue=ignoreMaskedValue, calcCompareSimpleGCR=calcCompareSimpleGCR, doMax=doMax, doPenalty=doPenalty, penaltyMode=penaltyMode, useRM=useRM,addOne=addOne, order = order, step1 = step1, step2 = step2, do3DData=do3DData, do2DData=do2DData,doGTM=doGTM)
                
                for vi in range(len(trainLocal[0][6][0])):
                    resultDict['rMS'][reductionName]['globalGCRAUC'][-1][vi][bestPerformanceIndex] = trainLocal[0][6][0][vi]
                    resultDict['rMA'][reductionName]['globalGCRAUC'][-1][vi][bestPerformanceIndex] = trainLocal[1][6][0][vi]
                resultDict['rMS'][reductionName]['globalGCRAUCReduction'][-1][bestPerformanceIndex]= trainLocal[0][6][1][2]
                resultDict['rMA'][reductionName]['globalGCRAUCReduction'][-1][bestPerformanceIndex] = trainLocal[1][6][1][2]


                resultDict['rMS'][reductionName]['globalGCRAUCGini'][-1][bestPerformanceIndex].append(trainLocal[0][5])
                resultDict['rMA'][reductionName]['globalGCRAUCGini'][-1][bestPerformanceIndex].append(trainLocal[1][5])
                for l in range(len(trainLocal[0][6][0][0])-1):
                    resultDict['rMS'][reductionName]['globalGCRAUCGini'][-1][bestPerformanceIndex].append(-1)
                    resultDict['rMA'][reductionName]['globalGCRAUCGini'][-1][bestPerformanceIndex].append(-1)
                resultDict['rMS'][reductionName]['globalGCRAUCConfidence'][-1][bestPerformanceIndex]=trainLocal[0][6][4]
                resultDict['rMA'][reductionName]['globalGCRAUCConfidence'][-1][bestPerformanceIndex]=trainLocal[1][6][4]

                thresholdPercents = np.array(list(range(0, 100, 20))) /100
                bestAUCtGCRIndex = np.argmax(np.array(resultDict['rMS'][reductionName]['globalGCRAUC'][-1][0][bestPerformanceIndex]))

                rMG = trainLocal[5] 
                if bestAUCtGCRIndex != 0:
                    lowestScore = np.min(rMG)
                    highestScore = np.max(rMG)
                    thresholds = np.zeros(len(thresholdPercents))
                    for ti, tp in enumerate(thresholdPercents):
                        thresholds[ti] = (highestScore-lowestScore) * tp + lowestScore


                    tindex = rMG >= thresholds[bestAUCtGCRIndex]
                    rMG = rMG * tindex
                    rMGGini = fcamQualityMetric(rMG)


                    resultDict['rMS'][reductionName]['globalGCRAUCGini'][-1][bestPerformanceIndex][bestAUCtGCRIndex] = rMGGini


                rMG = trainLocal[6]
                bestAUCtGCRIndex = np.argmax(np.array(resultDict['rMA'][reductionName]['globalGCRAUC'][-1][0][bestPerformanceIndex]))

                if bestAUCtGCRIndex != 0:
                    lowestScore = np.min(rMG)
                    highestScore = np.max(rMG)
                    thresholds = np.zeros(len(thresholdPercents))
                    for ti, tp in enumerate(thresholdPercents):
                        thresholds[ti] = (highestScore-lowestScore) * tp + lowestScore


                    tindex = rMG >= thresholds[bestAUCtGCRIndex]
                    rMG = rMG * tindex
                    rMGGini = fcamQualityMetric(rMG)


                    resultDict['rMA'][reductionName]['globalGCRAUCGini'][-1][bestPerformanceIndex][bestAUCtGCRIndex] = rMGGini


                for k in GCRPlus.gtmReductionStrings():
                    for vi in range(len(trainLocal[2][k]['aucScoreGlobal'])):
                        resultDict[k][reductionName]['globalGCRAUC'][-1][vi][bestPerformanceIndex].append(trainLocal[2][k]['aucScoreGlobal'][vi])
                    resultDict[k][reductionName]['globalGCRAUCReduction'][-1][bestPerformanceIndex].append(trainLocal[2][k]['aucScoreGlobalReduction'])
                    resultDict[k][reductionName]['globalGCRAUCGini'][-1][bestPerformanceIndex] = trainLocal[2][k]['giniScore']
                    resultDict[k][reductionName]['globalGCRAUCConfidence'][-1][bestPerformanceIndex]= trainLocal[2][k]['confidence']

                    rMG = trainLocal[3][k]
                    bestAUCtGCRIndex = np.argmax(np.array(resultDict[k][reductionName]['globalGCRAUC'][-1][0][bestPerformanceIndex]))

                    if bestAUCtGCRIndex != 0:
                        lowestScore = np.min(rMG)
                        highestScore = np.max(rMG)
                        thresholds = np.zeros(len(thresholdPercents))
                        for ti, tp in enumerate(thresholdPercents):
                            thresholds[ti] = (highestScore-lowestScore) * tp + lowestScore


                        tindex = rMG >= thresholds[bestAUCtGCRIndex]
                        rMG = rMG * tindex
                        rMGGini = gtmQualityMetric(rMG)


                        resultDict['rMA'][reductionName]['globalGCRAUCGini'][-1][bestPerformanceIndex][bestAUCtGCRIndex] = rMGGini





def do3DGCR(fullAttention, ix_train, trainLables, ix_test, testLables, num_of_classes, symbolsCount, trainCombis, ranges, rangesSmall, testCombis, gtmRange, reductionName="MixedClasses", mode=0, threshold=-1, doMetrics=False, calcCompareSimpleGCR=True, addMaskedValue=False,  ignoreMaskedValue=False, doMax=False, doPenalty=False, penaltyMode="entropy", addOne=False, useRM = True, order = 'lh', step1 = 'sum', step2 = 'sum', do3DData=True, do2DData=False, doGTM=True, doSoloGini=False, batchSize=6000):    

    if(order == 'lh'):
        axis1 = 0
        axis2 = 1
    elif(order == 'hl'):
        axis1 = 2
        axis2 = 0
    else:
        axis1 = 0
        axis2 = 1

    if do3DData:
        fullAttention = helper.doCombiStep(step1, fullAttention, axis1)
        fullAttention = helper.doCombiStep(step2, fullAttention, axis2)
        do3DData = False
        do2DData = True

    valuesA = helper.getMapValues(symbolsCount)
    if addMaskedValue:
        valuesA.append(-2)
    attentionQ = [[],[],np.array(fullAttention).squeeze()]
    print("Attention shape")
    print(np.array(fullAttention.squeeze()).shape)
    print(do3DData)
    print(do2DData)


    predictions = np.argmax(testLables,axis=1)
    gtmAbstractions = GCRPlus.gtmReductionStrings()

    print('start making attention')

    printCheck('gcrOut.txt', 'Start making Attention')
    if doMetrics:
        rMA, rMS, rM, gtms, gtmRM = GCRPlus.fastMakeAttention(attentionQ, ix_train, trainLables, trainCombis, ranges, order, step1, step2, num_of_classes, valuesA, makeSimpleToo=False, mode=mode, threshold=threshold, ignoreMaskedValue=ignoreMaskedValue, reductionName=reductionName, doMax=doMax, doPenalty=doPenalty, penaltyMode=penaltyMode, addOne=False, do3DData=do3DData, do2DData=do2DData, batchSize=batchSize)
        
        printCheck('gcrOut.txt', 'making done, starting AUC')
        rMAAUCglobal = fcamAUCMetric(rMA, rM, valuesA, ix_test, rangesSmall, testCombis, predictions, percentSteps=20,calcCompareSimpleGCR=calcCompareSimpleGCR, rMA=True)

        print('#################################')
        print(predictions)
        rMSAUCglobal =fcamAUCMetric(rMS, rM, valuesA, ix_test, rangesSmall, testCombis, predictions, percentSteps=20, calcCompareSimpleGCR=calcCompareSimpleGCR, rMA=False)
        if calcCompareSimpleGCR:
            gcrSFcamOutA = [rMAAUCglobal[0][0][0],rMAAUCglobal[0][1][0],rMAAUCglobal[0][2][0],rMAAUCglobal[0][3][0],rMAAUCglobal[0][4][0],rMAAUCglobal[0][5][0],rMAAUCglobal[0][6][0],rMAAUCglobal[0][7][0]],[rMAAUCglobal[1][0][0],rMAAUCglobal[1][1][0]], rMAAUCglobal[2][0],rMAAUCglobal[3][0],rMAAUCglobal[4][0]

            gcrSFcamOutS = [rMSAUCglobal[0][0][0],rMSAUCglobal[0][1][0],rMSAUCglobal[0][2][0],rMSAUCglobal[0][3][0],rMSAUCglobal[0][4][0],rMSAUCglobal[0][5][0],rMSAUCglobal[0][6][0],rMSAUCglobal[0][7][0]],[rMSAUCglobal[1][0][0],rMSAUCglobal[1][1][0]], rMSAUCglobal[2][0],rMSAUCglobal[3][0],rMSAUCglobal[4][0]
        else:
            gcrSFcamOutA = [rMAAUCglobal[0][0][0],rMAAUCglobal[0][1][0],rMAAUCglobal[0][2][0],rMAAUCglobal[0][3][0]],[rMAAUCglobal[1][0][0],rMAAUCglobal[1][1][0]], rMAAUCglobal[2][0],rMAAUCglobal[3][0],rMAAUCglobal[4][0]

            gcrSFcamOutS = [rMSAUCglobal[0][0][0],rMSAUCglobal[0][1][0],rMSAUCglobal[0][2][0],rMSAUCglobal[0][3][0]],[rMSAUCglobal[1][0][0],rMSAUCglobal[1][1][0]], rMSAUCglobal[2][0],rMSAUCglobal[3][0],rMSAUCglobal[4][0]


    else:

        if calcCompareSimpleGCR:
            rMA, rMS, rM, gtms, gtmRM, simpleRMA, simpleRMS, simpleGTMs = GCRPlus.fastMakeAttention(attentionQ, ix_train, trainLables, trainCombis, ranges, order, step1, step2, num_of_classes, valuesA, makeSimpleToo=calcCompareSimpleGCR, mode=mode, threshold=threshold, ignoreMaskedValue=ignoreMaskedValue, reductionName=reductionName, doMax=doMax, doPenalty=doPenalty, penaltyMode=penaltyMode, addOne=False, do3DData=do3DData, do2DData=do2DData, batchSize=batchSize)
            minScoresA = GCRPlus.calcFCAMMinScoreNP(simpleRMA)
            minScoresS = GCRPlus.calcFCAMMinScoreNP(simpleRMS)
            simpleRMAAUCglobal = GCRPlus.classFullAttFast(simpleRMA, ix_test, rangesSmall, testCombis, predictions, rM, minScoresA, valuesA, useRM=useRM, doPenalty=False, calcCompareSimpleGCR=False, rMA=True)
            simpleRMSAUCglobal = GCRPlus.classFullAttFast(simpleRMS, ix_test, rangesSmall, testCombis, predictions, rM, minScoresS, valuesA, useRM=useRM, doPenalty=False, calcCompareSimpleGCR=False, rMA=False)

        else:
            rMA, rMS, rM, gtms, gtmRM = GCRPlus.fastMakeAttention(attentionQ, ix_train, trainLables, trainCombis, ranges, order, step1, step2, num_of_classes, valuesA, makeSimpleToo=calcCompareSimpleGCR, mode=mode, threshold=threshold, ignoreMaskedValue=ignoreMaskedValue, reductionName=reductionName, doMax=doMax, doPenalty=doPenalty, penaltyMode=penaltyMode, addOne=False, do3DData=do3DData, do2DData=do2DData, batchSize=batchSize)

        printCheck('gcrOut.txt', 'making done')
        minScoresA = GCRPlus.calcFCAMMinScoreNP(rMA)
        minScoresS = GCRPlus.calcFCAMMinScoreNP(rMS)
        rMAAUCglobal = GCRPlus.classFullAttFast(rMA, ix_test, rangesSmall, testCombis, predictions, rM, minScoresA, valuesA, useRM=useRM, doPenalty=False, calcCompareSimpleGCR=False, rMA=True)
        rMSAUCglobal = GCRPlus.classFullAttFast(rMS, ix_test, rangesSmall, testCombis, predictions, rM, minScoresS, valuesA, useRM=useRM, doPenalty=False, calcCompareSimpleGCR=False, rMA=False)

        if calcCompareSimpleGCR:
            gcrSFcamOutA = [rMAAUCglobal[0][0][0],rMAAUCglobal[0][1][0],rMAAUCglobal[0][2][0],rMAAUCglobal[0][3][0],simpleRMAAUCglobal[0][0][0],simpleRMAAUCglobal[0][1][0],simpleRMAAUCglobal[0][2][0],simpleRMAAUCglobal[0][3][0]],[rMAAUCglobal[1][0][0],rMAAUCglobal[1][1][0]], rMAAUCglobal[2][0],rMAAUCglobal[3][0],rMAAUCglobal[4][0], 

            gcrSFcamOutS = [rMSAUCglobal[0][0][0],rMSAUCglobal[0][1][0],rMSAUCglobal[0][2][0],rMSAUCglobal[0][3][0],simpleRMSAUCglobal[0][0][0],simpleRMSAUCglobal[0][1][0],simpleRMSAUCglobal[0][2][0],simpleRMSAUCglobal[0][3][0]],[rMSAUCglobal[1][0][0],rMSAUCglobal[1][1][0]], rMSAUCglobal[2][0],rMSAUCglobal[3][0],rMSAUCglobal[4][0]
        else:
            gcrSFcamOutA = [rMAAUCglobal[0][0][0],rMAAUCglobal[0][1][0],rMAAUCglobal[0][2][0],rMAAUCglobal[0][3][0]],[rMAAUCglobal[1][0][0],rMAAUCglobal[1][1][0]], rMAAUCglobal[2][0],rMAAUCglobal[3][0],rMAAUCglobal[4][0]

            gcrSFcamOutS = [rMSAUCglobal[0][0][0],rMSAUCglobal[0][1][0],rMSAUCglobal[0][2][0],rMSAUCglobal[0][3][0]],[rMSAUCglobal[1][0][0],rMSAUCglobal[1][1][0]], rMSAUCglobal[2][0],rMSAUCglobal[3][0],rMSAUCglobal[4][0]


    minScoresA = GCRPlus.calcFCAMMinScoreNP(rMA)
    minScoresS = GCRPlus.calcFCAMMinScoreNP(rMS)
    printCheck('gcrOut.txt', 'gini start')
    print('start Gini')
    if doMetrics or doSoloGini:
        rMAGini = fcamQualityMetric(rMA)
        rMSGini = fcamQualityMetric(rMS)
    else:
        rMAGini = -1
        rMSGini = -1
    print('end gini')

    printCheck('gcrOut.txt', 'gini end')

    gcrGTMOut = dict()
    if doGTM:
        for i,e  in enumerate(gtmAbstractions):
            gcrGTMOut[e] = dict()
            minScoresGTM = GCRPlus.calcGTMMinScoreNP(gtms[e])
            if doMetrics:
                gcrGTMOut[e]['giniScore']  = gtmQualityMetric(gtms[e])
                rMSAUCglobalGTM =  gtmAUCMetric(gtms, e, ix_test, minScoresGTM, predictions, gtmRange, gtmRM, percentSteps=20, reductionName=reductionName,calcCompareSimpleGCR=calcCompareSimpleGCR)
                gcrGTMOut[e]['confidence']  = rMSAUCglobalGTM[4][0]

                gcrGTMOut[e]['aucScoreGlobal'] = rMSAUCglobalGTM[0]#[0]

                gcrGTMOut[e]['aucScoreGlobalReduction'] = rMSAUCglobalGTM[1][2]

                if calcCompareSimpleGCR:
                    gcrGTMOut[e]['performance'] = [rMSAUCglobalGTM[0][0][0],rMSAUCglobalGTM[0][1][0],rMSAUCglobalGTM[0][2][0],rMSAUCglobalGTM[0][3][0],rMSAUCglobalGTM[0][4][0],rMSAUCglobalGTM[0][5][0],rMSAUCglobalGTM[0][6][0],rMSAUCglobalGTM[0][7][0]]
                else:
                    gcrGTMOut[e]['performance'] = [rMSAUCglobalGTM[0][0][0],rMSAUCglobalGTM[0][1][0],rMSAUCglobalGTM[0][2][0],rMSAUCglobalGTM[0][3][0]]

            else:
                gtmOut =  GCRPlus.calcFullAbstractAttentionFast(gtms, e, ix_test, minScoresGTM, gtmRange, predictions, gtmRM, calcCompareSimpleGCR=False)

                if calcCompareSimpleGCR:
                    simpleGtmOut =  GCRPlus.calcFullAbstractAttentionFast(simpleGTMs, e, ix_test, minScoresGTM, gtmRange, predictions, gtmRM, calcCompareSimpleGCR=False)
                    gcrGTMOut[e]['performance'] = [gtmOut[0][0][0],gtmOut[0][1][0],gtmOut[0][2][0],gtmOut[0][3][0],simpleGtmOut[0][0][0],simpleGtmOut[0][1][0],simpleGtmOut[0][2][0],simpleGtmOut[0][3][0]]
                else:
                    gcrGTMOut[e]['performance'] = [gtmOut[0][0][0],gtmOut[0][1][0],gtmOut[0][2][0],gtmOut[0][3][0]]
                gcrGTMOut[e]['confidence']  = gtmOut[4][0]

                if doSoloGini:
                    gcrGTMOut[e]['giniScore']  = gtmQualityMetric(gtms[e])


    if doMetrics:
        return gcrSFcamOutS + (rMSGini, rMSAUCglobal), gcrSFcamOutA + (rMAGini, rMAAUCglobal), gcrGTMOut, gtms, rM, rMA, rMS
    elif doSoloGini:
        return gcrSFcamOutS + (rMSGini,), gcrSFcamOutA + (rMAGini,), gcrGTMOut, gtms, rM, rMA, rMS
    else: 
        return gcrSFcamOutS, gcrSFcamOutA, gcrGTMOut, gtms, rM, rMA, rMS

def do2DGCR(fullAttention, traindata, trainLables, testdata, testLables, num_of_classes, symbolsCount, do3DData=False, do3rdStep=False): 
    fullAttention = reduceMap(fullAttention, do3DData=do3DData, do3rdStep=do3rdStep)   
    valuesA = helper.getMapValues(symbolsCount)
    gtmS = dict()
    gtmA = dict()
    rm = dict()
    for lable in range(num_of_classes):
        gtmS[lable] = dict()
        gtmA[lable] = dict()
        rm[lable] = dict()
        for symbol in valuesA:
            gtmS[lable][symbol] = np.zeros(len(fullAttention[0]))
            gtmA[lable][symbol] = np.zeros(len(fullAttention[0]))
            rm[lable][symbol] = np.zeros(len(fullAttention[0]))

    traindataS = traindata.squeeze()
    testdata = testdata.squeeze()
    for i, y in enumerate(trainLables):
        for j, x in enumerate(traindataS[i]):
            gtmS[y][x][j] += fullAttention[i][j]
            rm[y][x][j] += +1

    for lable in range(num_of_classes):
        for symbol in valuesA:
            gtmA[lable][symbol] =  np.nan_to_num(gtmS[lable][symbol]/rm[lable][symbol])


    [rA, _,_,_], _, _,_,_ = evalGTM(gtmA, symbolsCount, testdata, testLables)
    [rS, _,_,_], _, _,_,_ = evalGTM(gtmS, symbolsCount, testdata, testLables) 

    
    return gtmS, gtmA, rm, rA, rS



def evalGTM(gtm, symbolsCount, testdata, testLables):
    valuesA = helper.getMapValues(symbolsCount)
    results = []
    predictResults = []
    biggestScores = []
    allLableScores = []

    maxScores = dict()
    
    for lable in gtm.keys():
        maxScores[lable] =  np.sum(np.max(list(gtm[lable].values()), axis=0))

    answers = []
    for ti in range(len(testdata)):
        answers.append(classifyGTM(testdata[ti], testLables[ti], gtm, maxScores))
        
    asynLabels = []
    for ans in answers:
        results.append(ans[1])
        predictResults.append(ans[2])
        biggestScores.append(ans[3])
        allLableScores.append(ans[0])
        asynLabels.append(ans[4])

    acc = metrics.accuracy_score(predictResults, asynLabels)
    predicsion = metrics.precision_score(predictResults, asynLabels, average='macro')
    recall = metrics.recall_score(predictResults, asynLabels, average='macro')
    f1= metrics.f1_score(predictResults, asynLabels, average='macro')

    confidenceAcc = helper.confidenceGCR(biggestScores, results)

    return [acc, predicsion, recall, f1], [allLableScores, biggestScores], results, predictResults, confidenceAcc


def classifyGTM(trial, ylabel, rMG, maxScores):
    lableScores = dict()

    for lable in rMG.keys():
        lableScores[lable] = 0
    
    for toVi in range(len(trial)):
        toV = trial[toVi]

        for lable in rMG.keys():
            lableScores[lable] += rMG[lable][float(toV)][toVi] 

    #get final score
    for lable in rMG.keys():
        lableScores[lable] = lableScores[lable]/maxScores[lable]

    #classification
    biggestLable = list(lableScores.keys())[np.argmax(list(lableScores.values()))]
    biggestValue = lableScores[biggestLable]
    boolResult = biggestLable == ylabel

    return lableScores, boolResult, biggestLable, biggestValue, ylabel
        
    
def getSubFCAM(gMA, valuesA, indexStart, indexEnd):
    rMA = GCRPlus.nested_dict_static()
    for lable in gMA.keys():
        for fromL in valuesA:
            for toL in valuesA:
                rMA[lable][fromL][toL] = np.zeros( indexEnd-indexStart, indexEnd-indexStart)
                for i in range(indexStart,indexEnd):
                    for j in range(indexStart,indexEnd):
                        rMA[lable][fromL][toL][i-indexStart][j-indexStart] = gMA[lable][fromL][toL][i][j]
    return rMA                        

def getSubGTM(gMA, valuesA, indexStart, indexEnd):
    stm = GCRPlus.nested_dict_static()
    for lable in gMA.keys():
        for redStr in GCRPlus.gtmReductionStrings():
            for fromL in valuesA:
                stm[lable][redStr][fromL] = np.zeros(indexEnd-indexStart)
                for i in range(indexStart,indexEnd):
                    stm[lable][redStr][fromL][i-indexStart] = gMA[lable][redStr][fromL][i]
    return stm


def getLasaThresholds(saliencyMap, data, thresholdFactor, do3DData=False, do3rdStep=False, globalT=True, axis1= 2, axis2=0, axis3=1, op1='sum',op2='sum',op3='sum'):
    if do3DData:
        saliencyMap = helper.doCombiStep(op1, saliencyMap, axis1)
        saliencyMap = helper.doCombiStep(op2, saliencyMap, axis2) 
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 
    elif do3rdStep:
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 

    X_sax = np.array(data).squeeze()
    heats = saliencyMap.squeeze()


    if globalT:
        lowestScore = np.min(saliencyMap)
        highestScore = np.max(saliencyMap)
    else:
        lowestScore = np.min(saliencyMap, axis=(1))
        highestScore = np.max(saliencyMap, axis=(1))

    return (highestScore-lowestScore) * thresholdFactor + lowestScore

def doSimpleLasaReduction(saliencyMap, data, thresholdFactor, trainCombis, saliencyCombis, doFidelity=False, do3DData=False, do3rdStep=False, globalT=True, processCombis=True, maskValue=-2, axis1= 2, axis2=0, axis3=1, op1='sum',op2='sum',op3='sum'):
    print('new ROAR start')
    newX = []
    reduction = []

    if do3DData:
        saliencyMap = helper.doCombiStep(op1, saliencyMap, axis1)
        saliencyMap = helper.doCombiStep(op2, saliencyMap, axis2) 
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 
    elif do3rdStep:
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 

    X_sax = np.array(data).copy()
    heats = saliencyMap.squeeze()


    if globalT:
        lowestScore = np.min(saliencyMap)
        highestScore = np.max(saliencyMap)
    else:
        lowestScore = np.min(saliencyMap, axis=(1))
        highestScore = np.max(saliencyMap, axis=(1))

    threshold = (highestScore-lowestScore) * thresholdFactor + lowestScore
    if not globalT:
        threshold = threshold[:,None]

    saliencyCombis = np.array(saliencyCombis)
    threshold = np.array(threshold)
    newTrainCombis = trainCombis.copy()       
    if doFidelity:
        X_sax = X_sax.flatten()
        X_sax[saliencyMap > threshold] = maskValue

        if processCombis:         
            if globalT:
                newTrainCombis[:, 1:][saliencyCombis[:,1:] > threshold] = maskValue
            else:
                nShape =  saliencyCombis[:,1:].shape
                tholds = (saliencyCombis[:,1:].reshape((len(threshold), -1)) > threshold).reshape(nShape)
                newTrainCombis[:, 1:][tholds] = maskValue

    else:
        X_sax = X_sax.squeeze()
        X_sax[saliencyMap < threshold] = maskValue

        if processCombis:
               
            
            if globalT:
                newTrainCombis[:, 1:][saliencyCombis[:,1:] < threshold] = maskValue
            else:
                nShape =  saliencyCombis[:,1:].shape

                tholds = (saliencyCombis[:,1:].reshape((len(threshold), -1)) < threshold).reshape(nShape)

                newTrainCombis[:, 1:][tholds] = maskValue


    reduction =  np.sum(X_sax == maskValue) / len(X_sax.flatten())


    return X_sax, reduction, newTrainCombis

def doSimpleLasaROAR(saliencyMap, data, threshold, doBaselineT=False, doFidelity=False, do3DData=False, do3rdStep=False, axis1= 2, axis2=0, axis3=1, op1='max',op2='sum',op3='max'):
    print('new ROAR start')
    newX = []
    reduction = []

    if do3DData:
        saliencyMap = helper.doCombiStep(op1, saliencyMap, axis1)
        saliencyMap = helper.doCombiStep(op2, saliencyMap, axis2) 
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 
    elif do3rdStep:
        saliencyMap = helper.doCombiStep(op3, saliencyMap, axis3) 

    X_sax = np.array(data).squeeze()
    heats = saliencyMap.squeeze()
    heats = preprocessing.minmax_scale(heats, axis=1)


    if doBaselineT:
        cutOff = threshold
        threshold = np.max(np.array(heats)[:,-1 * cutOff:], axis=1)

    for index in range(len(saliencyMap)):
                
            X_ori = X_sax[index]
            heat = heats[index] 


        
            if doBaselineT:
                borderHeat = threshold[index]
            else:
                maxHeat = np.max(heat)
                borderHeat = maxHeat*threshold
        
            fitleredSet = []
            skips = 0 
            for h in range(len(heat)):
                if validataHeat(heat[h], borderHeat, doFidelity):
                    fitleredSet.append(X_ori[h])
                else:
                    fitleredSet.append(-2)
                    skips += 1

            reduction.append(skips/len(heat))
            newX.append([fitleredSet])

    newX = np.array(newX, dtype=np.float32)
    newX = np.moveaxis(newX, 1,2)

    return newX, reduction


def validataHeat(value, heat, doFidelity):
    if doFidelity:
        return value <= heat
    else:
        return value >= heat