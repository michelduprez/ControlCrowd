from __future__ import division
import numpy as np
import scipy.integrate as integrate
from scipy import optimize
import math


def minimal_time(t_vector,s_vector):
	t = np.sort(t_vector,axis=0) # increasing order
	s = np.sort(s_vector,axis=0)[::-1,:] # decreasing order
	#print(t)
	#print(s)
	return max(t+s)[0,0]





def cost(x0,t0,x1,t1):
	infinity = math.inf
	if t0<t1:
		    c = (x0-x1)**2+(t0-t1)**2
	else:
		    c = infinity
	return c



def time_in_omega(velocity,mu_h,a,b,delta_x,dt):
	c = a + delta_x/2
	d = b - delta_x/2

	# initialisation of the final points and time
	y = np.zeros((len(mu_h),1))
	t = np.zeros((len(mu_h),1))

	for i in range(len(mu_h)):
		yn = mu_h[i,0]
		tn = 0.0
		while (yn<c or yn>d):
			k1 = velocity(yn,tn)
			k2 = velocity(yn+dt*k1/2,tn+dt/2)
			k3 = velocity(yn+dt*k2/2,tn+dt/2)
			k4 = velocity(yn+dt*k3,tn+dt)
			yn = yn + (dt/6)*(k1+2*k2+2*k3+k4)
			tn = tn + dt
		y[i,0] = yn
		t[i,0] = tn
	return np.matrix(y),np.matrix(t)


def permutation(x0_array,t0_array,x1_array,t1_array):
	#print(len(x0_array),len(t0_array),len(x1_array),len(t1_array))
	#print(t0_array[20,0])
	nb_elt = len(x0_array)
	# construction of the constraint matrix for the minimisation
	A_eq = np.zeros((2*nb_elt,nb_elt**2))
	for i in range(nb_elt):
		for j in range(nb_elt):
			A_eq[i,j+i*nb_elt] = 1.0

	for i in range(nb_elt):
		for j in range(nb_elt):
			A_eq[i+nb_elt,(j-1)*nb_elt+i] = 1.0
	b_eq = np.ones(2*nb_elt)


	# computation of the different cost
	c_mat = np.zeros((nb_elt,nb_elt))
	for i in range(nb_elt):
		for j in range(nb_elt):
			c_mat[i,j] = cost(x0_array[i,0],t0_array[i,0],x1_array[j,0],t1_array[j,0])
	c_list = np.reshape(c_mat,nb_elt**2)

	# optimization
	opt = optimize.linprog(c_list,A_eq=A_eq,b_eq=b_eq,bounds=(0,None), method='simplex')
	print(min(c_list))
	return np.matrix(np.round(np.reshape(opt.x,(nb_elt,nb_elt))))

#x0_array = np.array([0.0,2.0,1.0])
#t0_array = np.array([0.0,-1.0,1.0])
#x1_array = np.array([4.0,5.0,2.0])
#t1_array = np.array([1.0,2.0,0.0])

#res = cost(x0_array,t0_array,x1_array,t1_array)
#print(cost(0.0,0.0,1.0,1.0))



