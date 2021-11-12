import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_auc(actual, preds):
    fpr, tpr, thresholds = roc_curve(actual, preds[:, 1])
    plt.plot(fpr, tpr, 'r')
    plt.plot([0,1],[0,1],'b')
    plt.title('AUC: {}'.format(auc(fpr, tpr)))
    plt.show()