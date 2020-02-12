from __future__ import division
import numpy as np



def interpolate(mu_left,mu_right,mu_val,Ix):
	mu = np.zeros(len(Ix))
	#print(mu_left,mu_right,mu_val,len(mu_left))
	for i in range(len(mu)):
		for j in range(np.size(mu_left)):
			if Ix[i]>=mu_left[0,j] and Ix[i]<mu_right[0,j]:
				mu[i]=mu[i]+mu_val[0,j]
	return mu
