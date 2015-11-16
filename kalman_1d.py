import numpy
import pylab
import scipy

# intial parameters
n_iter = 50
sz = (n_iter,) # size of array
y0 = 100  # initial value
g = 9.81
dt = 0.2
ytruth = numpy.zeros(sz)
for i in range(0,n_iter):
    ytruth[i] = y0 - g*i*dt - g*i*dt**2/2.
ymeas = numpy.random.normal(ytruth,2) # observations

Q = 0 #1e-5 # process variance

# allocate space for arrays
yhat=numpy.zeros(sz)      # a posteri estimate of x
yhatminus=numpy.zeros(sz) # a priori estimate of x

R = 1**2 # estimate of measurement variance, change to see effect

# intial guesses
yhat[0] = 100.
P = 1.

for k in range(1,n_iter):
    # time update
     yhatminus[k] =  yhat[k-1] - g * dt - g*dt**2/2.
     Pminus = P #+Q

     # measurement update
     K = Pminus/( Pminus+R )
     yhat[k] = yhatminus[k]+K*(ymeas[k]-yhatminus[k])
     P = (1-K)*Pminus

pylab.figure()
pylab.plot(ytruth,'r-',color='g',label='truth value')
pylab.plot(ymeas,'k+',label='noisy measurements')
pylab.plot(yhat,'b-',label='a posteri estimate')
pylab.plot(yhatminus,'y-',label='a priori estimate')
pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Height')
pylab.show()
