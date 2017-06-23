import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import show
from numpy import loadtxt

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y) 		
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradient(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad

def predict(theta, X):  
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
  
#load the dataset
data = pd.read_csv('dataset.txt', header=None, names=['Total count', 'Spam words', 'Spam'])
data.head()
#load complete

#visualized dataset plot
positive = data[data['Spam'].isin([1])]  
negative = data[data['Spam'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Spam words'], positive['Total count'], s=50, c='b', marker='o', label='Spam')  
ax.scatter(negative['Spam words'], negative['Total count'], s=50, c='r', marker='x', label='Not Spam')  
ax.legend()  
ax.set_xlabel('Spam word count')  
ax.set_ylabel('Total word count')
#show has been commented for now...  
show()
#finished visualization

#sigmoid function display
"""nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(nums, sigmoid(nums), 'r') 
show()"""
#sigmoid display ends

#testing out cost function
data.insert(0, 'Ones', 1)
cols=data.shape[1]
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]
X = np.array(X.values)  
y = np.array(y.values)  
theta = np.zeros(3) 
#print X.shape
#print y.shape
#print theta.shape
#print "Cost=",cost(theta, X, y)
#finished testing cost function - working properly

#testing out gradient function
import scipy.optimize as opt  
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))  
#print "Gradient=",cost(result[0], X, y)
gr=cost(result[0],X,y)  #gr has been given the gradient value, later used for the spam detection down below...
"""print "Gradient value= ",gr
W=[0.48,0.69,0.38,0.95,0.50,0.65,0.71,0.82,0.75]
for i in W:
	if i>gr:
		print "%f Spam\n"%i
	else:
		print "%f Not spam\n"%i"""
#finished testing gradient function - working properly

#testing out accuracy function
theta_min = np.matrix(result[0])  
predictions = predict(theta_min, X)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
#print 'accuracy = {0}%'.format(accuracy)  
#finished testing - working properly

#doing some tests for spam detection from the model
ch=1
while ch==1:
	name=raw_input('Enter a filename: ')
	f=open(name,"r")
	g=open("nocspamw.txt","r")
	fcontent=f.read();
	gcontent=g.read();
	fwords=fcontent.split()
	gwords=gcontent.split()
	twc=0.0
	swc=0.0
	for i in fwords:
		twc=twc+1
	for i in fwords:
		for j in gwords:
			if i==j:
				swc=swc+1
	prob=float(swc/twc)
	print "Total word count= ",twc
	print "Spam word count= ",swc
	print "Probability obtained= ",prob
	if prob>gr:
		print "%s is spam\n"%name
	else:
		print "%s is not spam\n"%name
	ch=int(input('Wish to check more messages? (1-Yes/0-No): '))	
#tests finished - working properly 













