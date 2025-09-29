from tslearn.datasets import UCR_UEA_datasets
from sklearn.utils import shuffle
import numpy as np
from random import randint
from time import sleep
from pyts.datasets import ucr_dataset_list



def datasetSelector(dataset, seed_Value, number=0, doOverSampling=True, topLevel='and', symbols = 4, nrEmpty = 2, andStack = 1, orStack = 1, xorStack = 1, nrAnds = 2, nrOrs = 2, nrxor = 2, trueIndexes=[1,3], test_size=0.2, orOffSet=0, xorOffSet=0, redaundantIndexes = [[0,0]]):
    symbolCount = 0
    if dataset == 'UCR':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = doUCR(seed_Value, number, takeName = False)
        symbolCount = 0
    elif dataset == 'UCR-name':
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes = doUCR(seed_Value, number, takeName = True)
        symbolCount = 0
    else:
        X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, symbolCount = []

    y_train = np.array(y_train)
    y_train = y_train.astype(float)
    y_test = np.array(y_test)
    y_test = y_test.astype(float)
    X_test = np.array(X_test)
    X_test = X_test.astype(float)
    X_train = np.array(X_train)
    X_train = X_train.astype(float)   
    y_testy = np.array(y_testy)
    y_trainy = np.array(y_trainy)

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, dataName, num_of_classes, symbolCount


def doUCR(seed_value, number, takeName = True, retry=0, use_cache=True):
    try:
        datasets = UCR_UEA_datasets(use_cache=use_cache)
        dataset_list = univariate_equal_length 
        if takeName:
            print(str(number) + " WRONG#########")
            datasetName = number
        else:
            print(str(number) + " RIGHT#########")
            datasetName = dataset_list[number]
        
        X_train, y_trainy, X_test, y_testy = datasets.load_dataset(datasetName)
        
        setY = list(set(y_testy))
        setY.sort()
        print(setY)

        num_of_classes = len(set(y_testy))
        seqSize = len(X_train[0])

        X_train, y_trainy = shuffle(X_train, y_trainy, random_state = seed_value)

        y_train = []
        print(num_of_classes)
        for y in y_trainy:
            y_train_puffer = np.zeros(num_of_classes)
            y_train_puffer[setY.index(y)] = 1
            y_train.append(y_train_puffer)

        y_trainy = np.argmax(y_train,axis=1) +1 
            
        y_test = []
        for y in y_testy:
            y_puffer = np.zeros(num_of_classes)
            y_puffer[setY.index(y)] = 1
            y_test.append(y_puffer)
            
        y_testy = np.argmax(y_test,axis=1) +1 
    
    except Exception as e:
        print(e)
        if retry < 5:
            sleep(randint(10,30))

            if retry == 4:
                return doUCR(seed_value, number, takeName = takeName, retry=retry+1, use_cache=False)
            else:
                return doUCR(seed_value, number, takeName = takeName, retry=retry+1) 
        return [],[],[],[],[],[],99999,number, 0

    return X_train, X_test, y_train, y_test, y_trainy, y_testy, seqSize, datasetName, num_of_classes

# 112 equal length/no missing univariate time series classification problems
univariate_equal_length = [
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]
