import numpy as np

def best_accuracy_ml(labs, preds):
    log_and = np.logical_and(np.array(labs) == 1, np.array(preds) == 1)
    sum = np.sum(log_and, axis=1)
    return np.sum(sum >= 1) / len(preds)
