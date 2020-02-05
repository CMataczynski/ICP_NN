# ICP_NN

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
    |--[initial_dataset]
    |   |-4 csv's
    |--[RAW_dataset]
    |   |-provided dataset
--[experiments]
    |--[ModelName_ID]
    |   |--[model_weights]
    |   |   |-model_final.pth
    |   |   |-model_best.pth
    |   |-tensorboard events file
--[models]
    |-Files with code for the models
-utils.py:
    |-Dataset loader class
-training_loop.py:
    |-Training loop with tensorboard hooks and saver
-experiment_manager.py:
    |-Experiment management class, example of learning the network
-experiment_batch.py
    |-Multiple experiments management structure
```
## Running the learning process
```
python experiment_manager.py
```
or for batch learning
```
python experiment_batch.py
```
## Testing the models
TBD.
## Results
TBD.
