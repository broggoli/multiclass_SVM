import numpy as np
import matplotlib.pyplot as plt
import collections
import plot

class Support_vector_machine(object):
	"""A suport vector machine to classify linear separable Data"""

	def __init__(self, learning_rate=1, visual=True):
		"""initializes the SVM. option to visualize the Trained model"""
		self.visual = visual
		self.initial_lr = learning_rate
		self.epochs = 10000
		self.update_inter = 10
		self.learning_rate_cap = 10E-15

	def fit(self, X, Y, epochs=10000, X_test=[], Y_test=[], update_inter=10, learning_rate_cap=10E-15, Ws=[]):
		self.epochs = epochs
		self.update_inter = update_inter
		self.learning_rate_cap = learning_rate_cap
		self.classifiers = []
		cl = np.unique(Y)
		if len(cl) < 2:
			print("There must be at least 2 different classes to train on!")
		elif len(cl) >= 2:
			#loop through all the different features
			for positive_class in range(len(cl)):
				print(positive_class)
				labels = np.array([1 if l == positive_class else -1 for l in Y]).reshape((-1))
				test_labels = np.array([1 if l == positive_class else -1 for l in Y_test]).reshape((-1))
				#return self.fit_binary(X, labels, X_test=X_test, Y_test=test_labels)
				self.classifiers.append(self.fit_binary(X, labels, X_test=X_test, Y_test=test_labels))
			"""
			X_combined= np.vstack((X[:,0:2], X_test))
			Y_combined = np.hstack((Y, Y_test))

			plot.plot_decision_regions(X=X_combined, y=Y_combined, classifier=self)
			plt.xlabel('petal length [cm]')
			plt.ylabel('petal width [cm]')
			plt.legend(loc='upper left')
			plt.show()
			print(self.classifiers)"""


	"""Trains the Model"""
	def fit_binary(self, X, Y, W=[], X_test=[], Y_test=[]):
		
		#initialize the learning rate
		learning_rate = self.initial_lr
		n_features = len(X[0])
		#add a bias term to the feature matrix so the decision boundary does not necessarily have to go through the origin
		X = self.add_bias(X)
		log = {}

		if(W == []):
			W = np.ones(n_features+1) 

		#Dicitionary for saving the best Ws
		opt_dict = {}
		#Gradient decent
		for epoch in range(1, self.epochs):

			for i, x in enumerate(X):
				#print(epoch, W, i)
				"""
				If misclassification -> lossfunction = c(x,y,f(x)) = (1 - y * f(x))+
				prediction = f(x) = x * W
				regularisation term = lamda * |w|^2
				"""
				prediction = np.dot(X[i], W)

				#update suport Vector
				if (Y[i] * prediction) < 1:
					W = W + learning_rate * ( X[i] * Y[i] - 2.0 / epoch * W)
				else:
					W = W * (1 - learning_rate * 2.0 / epoch)

			#calculate the loss for this epoch
			loss = self.loss(X, Y, W)

			#update the learning rate
			if epoch > 1:
				smallest_loss = sorted(opt_dict)[0]
				if(loss < smallest_loss):
					opt_dict[loss] = W
				else:
					learning_rate *= 0.9999

					W = opt_dict[smallest_loss]
			else:
				opt_dict[loss] = W

			#update the log
			if epoch % self.update_inter == 1:
				log[epoch] = (self.accuracy(loss,X.shape[0]), loss)

			if epoch < 10:
				X_combined= np.vstack((X[:,0:2], X_test))
				Y_combined = np.hstack((Y, Y_test))
				plot.plot_decision_regions(X=X_combined, y=Y_combined, classifier=self, Ws=[W])
				plt.xlabel('petal length [cm]')
				plt.ylabel('petal width [cm]')
				plt.legend(loc='upper left')
				plt.show()


			if learning_rate < self.learning_rate_cap:
				self.print_info(epoch, self.loss(X, Y, W), "Learning rate too small!",W, X_test, Y_test)
				W = opt_dict[smallest_loss]
				break
			elif loss == 0:
				self.print_info(epoch, self.loss(X, Y, W), "Loss is 0!",W, X_test, Y_test)
				break

		log[epoch] = (self.accuracy(loss,X.shape[0]), loss)


		print(W)
		X_combined= np.vstack((X[:,0:2], X_test))
		Y_combined = np.hstack((Y, Y_test))
		plot.plot_decision_regions(X=X_combined, y=Y_combined, classifier=self, Ws=[W])
		plt.xlabel('petal length [cm]')
		plt.ylabel('petal width [cm]')
		plt.legend(loc='upper left')
		plt.show()

		
		if self.visual == True:
			plot.visualize_acc_loss(log)
			print(self.accuracy(loss,X.shape[0]))
		# Return the trained normal vector
		return list(W)

	def classify(self, features, add_b=True, Ws=[]):
		"""returns a prediction for a set of data"""
		if add_b: 
			features = self.add_bias(np.array(features))
		if Ws == []:
			Ws = self.classifiers

		classifications = []
		print(Ws)
		for W in np.array(Ws):
			classifications.append(np.sign(np.dot(features, W)))
		
		print(classifications)
		output = [0 for f in features]
		if len(classifications) >= 2:
			for cl, values in enumerate(classifications):
				print("CLASS:",cl)
				for i, value in enumerate(values):
					print(value, output[i])
					if value == 1:
						output[i] = cl
		else:
			output = classifications
		print(output)
		return np.array(output)

	def add_bias(self, features):
		#adds a bias column to the feature matrix
		return np.concatenate( (features,-1 * np.ones((features.shape[0], 1))), 1 )

	def loss(self, x, y, W, add_b=False):
		return sum(np.absolute(1 - y * np.sign(np.dot(x, W))))

	def save_model(self):
		pass

	def accuracy(self, test_loss, n_samples):
		missclassified_samples = test_loss/2.0

		return (1 - missclassified_samples / float(n_samples-1))*100.0

	def print_info(self, epoch, train_loss, finished_eraly, W, X_test=[], Y_test=[]):
		print(finished_eraly)
		print("Trained after %i epochs." % epoch)
		print("Train loss is: %i!" % train_loss)

		if(X_test != []):
			test_loss = self.loss(self.add_bias(X_test), Y_test, W, add_b=True)
			print("Test loss is: %i!" % test_loss)

			print("Test accuracy: %.3f percent" %(self.accuracy(test_loss, X_test.shape[0]+1)))





