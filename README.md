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
    |--[full_corrected_dataset]
    |   |--[train]
    |   |   |-files in the form Class_ID.csv
    |   |--[test]
    |   |   |-files in the form Class_ID.csv
    |--[full_dataset]
    |   |--[train]
    |   |   |-files in the form Class_ID.csv
    |   |--[test]
    |   |   |-files in the form Class_ID.csv
    |--[initial_dataset]
    |   |-4 csv's
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
-utils.py:
    |-Dataset loader class, other utility functions
-training_loop.py:
    |-Training loop with tensorboard hooks and saver
-experiment_manager.py:
    |-Experiment management class, example of learning the network
-experiment_batch.py
    |-Multiple experiments management structure
-reformat_dataset.py
    |-Script for transforming raw dataset into training and testing datasets with unique id's and mapping
-Testing.ipynb
    |-Notebook for model testing
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
python experiment_batch.py
```
## Testing the models
To test the results, open anaconda prompt installing jupyter-notebook
```
conda install jupyterlab
conda install ipywidgets
```
Then activate the extenstions by typing
```
jupyter nbextension enable --py widgetsnbextension
```
And run the notebook in the working directory of the project
```
cd working_directory
jupyter-notebook
```
And open the testing, tweaking the folder names to run the contents properly. The sliders will allow you to test all the models.
## Results
To see the results, go to [the project Google Drive](https://drive.google.com/drive/folders/1IFwF1pn_IJrCovk3XIw8etag5nVL8veH?usp=sharing), and download the experiments folder to the working directory of the project. Then in anaconda run:
```
cd working_directory
tensorboard --logdir experiments
```
And go to your browser typing (by default)
```
localhost:6006
```
To access the tensorboard. You can look up models there as well as their results or confusion matrices.

## Visualization
To visualize the classification and dataset analysis results, go to Visualisation.ipynb notebook, and run first, second and two last cells (one with loading the data and other with interactive displays). You will have to provide path to load the data from pickle, which can be found here:
https://drive.google.com/file/d/1Zfo3VuIwvpI9TCNbNxTbHMVZslMh1nls/view?usp=sharing
