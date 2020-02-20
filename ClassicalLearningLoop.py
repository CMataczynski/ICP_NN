from models.ClassicalModels import SVM, KNN, DiscriminantAnalysis, GaussianProcess, \
    GaussianNBClass, DecisionTree, RandomForest, AdaBoost
from sklearn.tree import DecisionTreeClassifier
from utils import Initial_dataset_loader, get_fourier_coeff
import os
import numpy as np
import pandas as pd



input_size = 180
output_sizes = [4, 5]
rootdir = os.path.join(os.getcwd(), 'experiments')
dataset = "full_splitted_dataset"
datasets = os.path.join(os.getcwd(), "datasets", dataset)
train_dataset_path = os.path.join(datasets, "train")
test_dataset_path = os.path.join(datasets, "test")
exp_name = "four_classic_split"
fulls = [False, True]
labels_table = [["T1", "T2", "T3", "T4"], ["T1", "T2", "T3", "T4", "A+E"]]
# ort = lambda x,y: np.polynomial.hermite_e.hermefit(x, y, 7)
ort = get_fourier_coeff
# ort = lambda x, y: np.polynomial.chebyshev.chebfit(x, y, 5)
for output_size, labels, full in zip(output_sizes, labels_table, fulls):
    train_dataset = Initial_dataset_loader(train_dataset_path, full=full, ortho=ort).get_dataset()
    test_dataset = Initial_dataset_loader(test_dataset_path, full=full, ortho=ort).get_dataset()
    dataframe_out = []
    models = [
        SVM(input_size, output_size, labels),
        KNN(input_size, output_size, labels, n_neighbors=7),
        KNN(input_size, output_size, labels, n_neighbors=5),
        DiscriminantAnalysis(input_size, output_size, labels),
        GaussianProcess(input_size, output_size, labels),
        GaussianNBClass(input_size, output_size, labels),
        DecisionTree(input_size, output_size, labels, criterion='entropy'),
        RandomForest(input_size, output_size, labels, criterion='entropy'),
        AdaBoost(input_size, output_size, labels, base_estimator=DecisionTreeClassifier(max_depth=5))
    ]
    for model in models:
        model.learn(train_dataset["data"].tolist(), train_dataset["id"].tolist())
        mean_score = model.get_score(test_dataset["data"].numpy(), test_dataset["id"].numpy())
        f1_score = model.get_score(test_dataset["data"].numpy(), test_dataset["id"].numpy(), f1_score=True)
        print(model)
        dataframe_out.append(model.get_results())
    dataframe_out = pd.DataFrame(dataframe_out, columns=["name", "params", "acc", "f1 score"])
    dataframe_out.to_csv(os.path.join(rootdir, "Classical_experiments", exp_name+str(len(labels))+"cls.csv"), sep=";",
                         decimal=',')