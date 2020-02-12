from __future__ import division
#import numpy as np
#import scipy.integrate as integrate
#from scipy import optimize


Animation = True
charac = False
screen = True

# number of cells in space for an uniform mesh
#Nx = 8
# number of cells in space follwing the mass
Nm = 1000


# step of time
dt = 0.002

# time of controllability = minimal_time + epsilon
epsilon = 0.1

# step for the uniform discretisation
delta_x = 0.02
step_m = 1.0/Nm



test_case = 3


if test_case ==2:
	a0 = 0
	b0 = 1.25
	a1 = 4
	b1 = 7
	a = 2
	b = 3

if test_case ==3:
	a0 = 0
	b0 = 2
	a1 = 7
	b1 = 11
	a = 5
	b = 6

# initial measure
def mu_0(x,test_case):
	if test_case == 1:
		if x>-10 and x<0:
			res = 1.0/10.0
		else:
			res = 0.0
	if test_case == 2:
		if (x>0 and x<0.75):
			res = 1.0
		elif (x>1 and x<1.25):
			res = 1.0
		else:
		    res = 0.0
	if test_case == 3:
		if (x>0 and x<2):
			res = 1.0/2
		else:
		    res = 0.0
	return res


# target 
def mu_1(x,test_case):
	if test_case == 1:
		if (x>3 and x<4) or (x>9 and x<10):
			res = 0.5

		else:
			res = 0
	if test_case == 2:
		if (x>7 and x<10) or (x>12 and x<13):
			res = 0.5

		else:
			res = 0
	if test_case == 3:
		if (x>7 and x<8) or (x>10 and x<11):
			res = 1.0/2

		else:
			res = 0
	return res


# velocity for the forward computation of the solution
def velocity(x,t):
	return 1.0

# velocity for the forward computation of the solution
def velocity2(x,t):
	return -1.0



