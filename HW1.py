import random
import numpy as np
#import matplotlib.pyplot as plt

# parameters
N = 100 # No. of training points
D = 2  # 2-dimension

# area between f & g
area = 0

cnt0 = 0
for irun in range(1000):
	# training data
	x1, x2 = np.zeros((N, 1)), np.zeros((N, 1))
	for iN in range(N):
	    x1[iN] = random.uniform(-1, 1)
	    x2[iN] = random.uniform(-1, 1)
	xtrain = np.c_[np.ones((N, 1)), x1, x2]

	# target function: passes through two points
	x11, x21 = random.uniform(-1, 1), random.uniform(-1, 1) # 1st point
	x12, x22 = random.uniform(-1, 1), random.uniform(-1, 1) # 2nd point
	x0 = np.arange(-1, 1, .1) # for plotting purpose
	y0 = (x22 - x21)/(x12 - x11) * (x0 - x11) + x21

	# target: expected output
	y = np.zeros(N)
	for iN in range(N):
	    f = (x22 - x21)/(x12 - x11) * (x1[iN] - x11) + x21
	    if f < x2[iN]: y[iN] = 1
	    elif f > x2[iN]: y[iN] = -1

	# # visualize
	# plt.plot(x0, y0)
	# plt.scatter(x1, x2)

	# weight vector
	w = np.zeros(D+1) # initially, all points mis-classified
	# estimated label through Perceptron
	yp = np.zeros(N) # initially all 0
	cnt = 0
	while np.all(yp == y) == False:
	    evlt = list(np.equal(yp, y))
	    iN = evlt.index(False)
	    w += y[iN] * xtrain[iN, :] # update the weight factor
	    yp = np.sign(w.dot(xtrain.T))
	    cnt += 1
	    
	    # # visualize
	    # g0 = -(w[0] + w[1]*x0)/w[2]
	    # plt.plot(x0, g0)
	    # for i in range(N):
	    #     plt.text(x1[i], x2[i], str(yp[i]))
	    # plt.pause(1)

	    # input('Press enter to continue')
    
	cnt0 += cnt # No. of iterations to converge

	# estimate the difference between f & g using Monte Carlo
	ntest = 1000
	count = 0
	for itest in range(ntest):
	    x1test, x2test = random.uniform(-1, 1), random.uniform(-1, 1)
	    # target
	    fx2 = x21 + (x22-x21)/(x12-x11) * (x1test-x11)
	    if fx2 < x2test: target = 1
	    elif fx2 > x2test: target = -1
	    else: target = 0
	    # estimate
	    estimate = np.sign(w.dot([1, x1test, x2test]))
	    if estimate != target: count += 1
	area += count / ntest

print(area / 1000)
print(cnt0 / 1000)