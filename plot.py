from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.1, Ws=[]):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    if Ws != []:
        Z = classifier.classify(np.array([xx1.ravel(), xx2.ravel()]).T, Ws=Ws)
    else:
        Z = classifier.classify(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(Z)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

def show_data(dataset, line=None):
    """plots the data given to it, provided it is two dimensional"""
    if len(dataset[0]) == 3:
        print("its 2D!")
    else:
        pass
    y = dataset[:,-1]
    X = dataset[:,0:2]
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    for inedx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(inedx),
                    edgecolor='black',
                    marker=markers[inedx], 
                    label=cl)
    plt.legend(loc="upper right")

    if line != None:
        pass

    plt.show()

def visualize_acc_loss(log):

        l = sorted(log.items()) # sorted by key, return a list of tuples

        e, al = zip(*l) # unpack a list of pairs into two tuples
        a=[l[0] for l in al]
        l=[l[1] for l in al]
        # Two subplots, the axes array is 1-d
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(e, l, marker="o")
        axarr[0].set_title('Loss')

        axarr[1].plot(e, a, marker="o")
        axarr[1].set_title('Accuracy')
        
        plt.show()