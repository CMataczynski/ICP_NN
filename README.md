# End-to-End Automatic MorphologicalClassification Method of Intracranial PressurePulse Waveforms Using Deep Learning

## Creating environment
To create anaconda environment for this project you need to open Anaconda Prompt, change working directory to project folder and run the script
```
conda env create -f environment.yml
```
## Folder Structure
```
[.] - folder
--[datasets]
    |--[full_dataset]
    |   |--[train]
    |   |   |-files in the form Class_ID.csv
    |   |--[test]
    |   |   |-files in the form Class_ID.csv
    |--[RAW_dataset]
    |   |-provided dataset
    |--[RAW_corrected_dataset]
    |   |-provided dataset
--[experiments]
    |--[ModelName_ID]
    |   |--[model_weights]
    |   |   |-model_final.pth
    |   |   |-model_best.pth
    |   |-tensorboard events file
--[models]
    |-Files with code for the models
--[signal_processing]
    |-Files with code for the initial processing of the ICP/ABP signals
-utils.py:
    |-Dataset loader class, other utility functions
-training_loop.py:
    |-Training loop with tensorboard hooks and saver
-experiment_manager.py:
    |-Experiment management class, example of learning the network
-multiexp.py
    |-Multiple experiments management structure
-reformat_dataset.py
    |-Script for transforming raw dataset into training and testing datasets with unique id's and mapping
```
## Create the dataset
You need raw dataset in the datasets folder.
Modify the code in the reformat dataset to suit your needs - best leave new dataset as full_corrected_dataset
Then run:
```
python reformat_dataset.py
```

## Running the learning process
```
python experiment_manager.py
```
or for batch learning
```
python multiexp.py
```
