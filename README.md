# SymbolicFeatureGCR

The repository provides the code for the final experiments of the dissertation "Analysing Higher-Level Features and Relations on Time Series Data via Attribution-based Explainability Methods"

The experiments show the advantage of symbolic higher-level features over feature extraction frameworks on time series data. They also include and evaluate the GCR as an XAI approach.

## Most important files for the experiments:

- symbolicFeaturesPreprocessing.py: Contains the preprocessing
- preprocessingData.yaml: Seml configuration for the preprocessing
- symbolicFeaturesExperiments.py: Contains the main experiments
- mixModel.yaml: Seml configuration for the main experiments
- BilderGCR contains the summarised results in the form of images and tables
- gcrResultResultProcessing.py: Evaluation script to generate all files in BilderGCR
- BilderGCR\general\fullPerformance.csv: Full list of our numeric performance results

## Dependencies and installation guide

A list of all needed dependencies (other versions might work but are not guaranteed to do so):
- python==3.9.19
- conda==22.9.0
- pip==23.3

We suggest a fresh conda environment! <br>
Create a new environment and run the following lines:<br>

Adapt CUDA versions if needed: <br>
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidiaor: <br>

Further install: <br>

pip install seml==0.3.7 captum==0.8.0 tslearn==0.6.3 scikit-learn==1.6.1 pandas==2.2.3 sktime==0.37.0 matplotlib==3.9.4 shap==0.47.2 grad-cam==1.5.5 joblib==1.4.2 dill==0.4.0 pycatch22==0.4.5 tsfel==0.1.9 tsfresh==0.21.0 pyts==0.13.0 transformer_encoder==0.0.3 imbalanced-learn==0.11.0 numpy==1.23.5


Note, sometimes a cv2 package is missing. In that case use: pip install opencv-python-headless <br>

### How to run (over SEML)

The experiments are set up to work with SEML on our cluster. Change the .yaml files for parameter tuning. <br>

1. Set up seml with seml configure <font size="6">(yes you need a mongoDB server for this and yes the results will be saved a in separate file, however seml does a really well job in managing the parameter combinations in combination with slurm) </font>
2. Configure the yaml files you want to run. Probably you only need to change the number of maximal parallel experiments ('experiments_per_job' and 'max_simultaneous_jobs') and the memory and cpu use ('mem' and 'cpus-per-task').
3. Add and start the seml experiment. For example like this:
	1. seml symbolicGCRPreprocessing add preprocessingData.yaml
	3. seml symbolicGCRPreprocessing start
4. Let the preprocessing finish
5. Add and start the seml experiment. For example like this:
	1. seml symbolicGCR add mixModel.yaml
	3. seml symbolicGCR start
6. Check with "seml simpleGCR status" till all your experiments are finished 
7. Please find the results in the pResultSymbolicGCR or filteredSymbolicGCR folder. The results can be evaluated using the gcrResultResultProcessing.py



## Reference

Transformer and LRP implementations are taken and adapted from https://github.com/hila-chefer/Transformer-Explainability <br>
The SHAP-IQ implementation is taken from https://github.com/mmschlk/shapiq
The Gini Index implementation is taken from https://github.com/oliviaguest/gini<br>

## Cite and publications

This code represents the used model for the following publication:<br>
"Analysing Higher-Level Features and Relations on Time Series Data via Attribution-based Explainability Methods" (TODO Link)

If you use, build upon this work or if it helped in any other way, please cite the linked publication.