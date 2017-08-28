import SVM
import plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cProfile

np.random.seed(10)
# Resources https://www.youtube.com/watch?v=7C9ptx68ADM
#import the Data with pandas

df = pd.read_csv("data/Iris.csv")
#shuffle the Data around
df = df.reindex(np.random.permutation(df.index))
"""prepare the data for the classifier"""

#take the first 100 data points and assign them either a -1 or a 1, depending on whether they are Iris-setosa or Iris-Versicolor
n_data = df.shape[0]
train_test_split = 0.3
n_train_data = int(n_data * 0.3)
labels = df.iloc[0:, 5].values
Y = [1 if label=="Iris-virginica" else 0 for label in labels]

"""
labels = df.iloc[0:, 5].values
features = np.unique(labels)
Y=[None for x in labels]
for i, f in enumerate(features):
	for x, label in enumerate(labels):
		if label==f : Y[x] = i
"""
Y_train, Y_test = Y[:n_train_data], Y[n_train_data:n_data]

#assign the imput data to X (only the rows 0 and 2 are chosen as features -> just so it's still 2D)
X_train = df.iloc[0 : n_train_data, [0,3]].astype(float).values
X_test = df.iloc[n_train_data: n_data, [0,3]].astype(float).values

classifier = SVM.Support_vector_machine()
print(Y_train)

#cProfile.run('SVM.Support_vector_machine(visual=False).fit(X_train,Y_train)')
W = classifier.fit_binary(X_train, Y_train, X_test=X_test, Y_test=Y_test)
print("W: " , W)
"""
X_combined= np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))

plot.plot_decision_regions(X=X_combined, y=Y_combined, classifier=classifier, test_idx=range(n_train_data, n_data))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()"""
