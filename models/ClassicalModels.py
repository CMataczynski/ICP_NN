import sklearn as skl
from utils import plot_confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class ClassicalModel:
    def __init__(self, input_size, output_size, labels, class_weights=None):
        self.input_size = input_size
        self.output_size = output_size
        self.labels = labels
        self.class_weights = class_weights
        self.model = None
        self.name = None
        self.mean_score = None
        self.f1_score = None

    def __str__(self):
        output_str = self.name + "\n input size: {}, \n output_size: {}, \n".format(self.input_size, self.output_size)
        if self.mean_score is not None:
            output_str += "Mean Score: {}%\n".format(self.mean_score*100)
        if self.f1_score is not None:
            output_str += "F1 Score: {}\n".format(self.f1_score)
        return output_str

    def get_results(self):
        return [self.name.split('\n')[0], self.name.split('\n')[1:], self.mean_score, self.f1_score]

    def learn(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_score(self, x, y, f1_score=False):
        if not f1_score:
            self.mean_score = self.model.score(x, y, self.class_weights)
            return self.mean_score
        else:
            predicted = self.predict(x)
            self.f1_score = skl.metrics.f1_score(y, predicted, average="weighted")
            return self.f1_score

    def get_confusion_matrix(self, x, y):
        return plot_confusion_matrix(y, self.predict(x), self.labels)


class SVM(ClassicalModel):
    def __init__(self, input_size, output_size, labels, class_weights=None, **kwargs):
        super().__init__(input_size, output_size, labels, class_weights)
        self.model = SVC(**kwargs)
        self.name = "SVM:\n" + str(self.model.get_params())


class KNN(ClassicalModel):
    def __init__(self, input_size, output_size, labels, class_weights=None, **kwargs):
        super().__init__(input_size, output_size, labels, class_weights)
        self.model = KNeighborsClassifier(**kwargs)
        self.name = "KNN:\n" + str(self.model.get_params())


class GaussianProcess(ClassicalModel):
    def __init__(self, input_size, output_size, labels, class_weights=None, **kwargs):
        super().__init__(input_size, output_size, labels, class_weights)
        self.model = GaussianProcessClassifier(**kwargs)
        self.name = "Gaussian Process Classifier: \n" + str(self.model.get_params())


class DecisionTree(ClassicalModel):
    def __init__(self, input_size, output_size, labels, class_weights=None, **kwargs):
        super().__init__(input_size, output_size, labels, class_weights)
        self.model = DecisionTreeClassifier(**kwargs)
        self.name = "Decision Tree Classifier: \n" + str(self.model.get_params())


class RandomForest(ClassicalModel):
    def __init__(self, input_size, output_size, labels, class_weights=None, **kwargs):
        super().__init__(input_size, output_size, labels, class_weights)
        self.model = RandomForestClassifier(**kwargs)
        self.name = "Random Forest: \n" + str(self.model.get_params())


class AdaBoost(ClassicalModel):
    def __init__(self, input_size, output_size, labels, class_weights=None, **kwargs):
        super().__init__(input_size, output_size, labels, class_weights)
        self.model = AdaBoostClassifier(**kwargs)
        self.name = "AdaBoost Classifier: \n" + str(self.model.get_params())


class GaussianNBClass(ClassicalModel):
    def __init__(self, input_size, output_size, labels, class_weights=None, **kwargs):
        super().__init__(input_size, output_size, labels, class_weights)
        self.model = GaussianNB(**kwargs)
        self.name = "Gaussian Naive Bayes Classifier: \n" + str(self.model.get_params())


class DiscriminantAnalysis(ClassicalModel):
    def __init__(self, input_size, output_size, labels, class_weights=None, **kwargs):
        super().__init__(input_size, output_size, labels, class_weights)
        self.model = QuadraticDiscriminantAnalysis(**kwargs)
        self.name = "Quadratic Discriminant Analysis: \n" + str(self.model.get_params())




