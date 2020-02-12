from __future__ import division
import numpy as np
import scipy.integrate as integrate
#import matplotlib.pyplot as plt
import scipy.interpolate
#from matplotlib import animation
from matplotlib.path import Path
import matplotlib.patches as patches
import math
from matplotlib import cm
#from scipy import optimize
#from cvxopt import solvers, matrix
import pulp

import matplotlib.pyplot as plt
import matplotlib.animation as animat


##############################
### begining of parameters ###
##############################

animation = True
charac = False
screen = False


number_frame = 3


# step of time
dt = 0.05

# time of controllability = minimal_time + epsilon
epsilon = 0.1


output = True


test_case = 3


if test_case == 1:
	a0 = 0 # minx for mu0
	b0 = 2 # maxx for mu0
	c0 = 0 # miny for mu0
	d0 = 1 # maxy for mu0
	a1 = 7 # minx for mu0
	b1 = 11 # maxx for mu0
	c1 = 0 # miny for mu0
	d1 = 1 # maxy for mu0
	a = 5 # minx for omega
	b = 6 # maxx for omega
	c = 0 # miny for omega
	d = 1 # maxy for omega
	Nm = 4 # number of cells in space follwing the mass
	# step for the uniform discretisation
	delta_x = 0.5
	step_m = 1.0/Nm


if test_case ==2:
	a0 = 0 # minx for mu0
	b0 = 2 # maxx for mu0
	c0 = 0 # miny for mu0
	d0 = 1 # maxy for mu0
	a1 = 5 # minx for mu0
	b1 = 8 # maxx for mu0
	c1 = 0 # miny for mu0
	d1 = 1 # maxy for mu0
	a = 3 # minx for omega
	b = 4 # maxx for omega
	c = 0 # miny for omega
	d = 1 # maxy for omega
	Nm = 12 # number of cells in space follwing the mass
	delta_x = 0.25	# step for the uniform discretisation
	step_m = 1/(np.sqrt(3)*4)

if test_case ==3 or test_case ==4:
	a0 = 0 # minx for mu0
	b0 = 4 # maxx for mu0
	c0 = 1 # miny for mu0
	d0 = 2 # maxy for mu0
	a1 = 8 # minx for mu0
	b1 = 14 # maxx for mu0
	c1 = 0 # miny for mu0
	d1 = 4 # maxy for mu0
	a = 5 # minx for omega
	b = 7 # maxx for omega
	c = 0 # miny for omega
	d = 4 # maxy for omega
	#Nm = 12 # number of cells in space follwing the mass
	delta_x = 1# step for the uniform discretisation
	step_m = 1/16


coeff_retr_x =0.0
coeff_retr_y =0.0
coeff_retr_final = 0.0#0.1

# initial measure
def mu_0(x,y,test_case):
	if test_case == 1:
		if (x>0 and x<2 and y>0 and y<1):
			res = 1.0/2
		else:
		    res = 0.0
	if test_case == 2:
		if (x>0 and x<2 and y>0 and y<0.25) or (x>0 and x<2 and y>0.75 and y<1.0) or (x>0 and x<1 and y>0 and y<1):
			res = 0.75
		else:
		    res = 0.0
	if test_case == 3 or test_case == 4:
		if (x>0 and x<4 and y>1 and y<3):
			res = 1.0/8.0
		else:
		    res = 0.0
	return res


# target 
def mu_1(x,y,test_case):
	if test_case == 1:
		if (x>7 and x<8 and y>0 and y<1) or (x>10 and x<11 and y>0 and y<1):
			res = 1.0/2
		else:
			res = 0.0
	if test_case == 2:
		if (x>5 and x<5.5 and y>0 and y<1) or (x>5.5 and x<6 and y>0 and y<0.25) or (x>5.5 and x<6 and y>0.75 and y<1.0) or (x>7 and x<7.5 and y>0 and y<1) or (x>7.5 and x<8 and y>0 and y<0.25) or (x>7.5 and x<8 and y>0.75 and y<1):
			res = 0.75
		else:
			res = 0.0
	if test_case == 3 or test_case == 4:
		if (x>8 and x<14 and y>0 and y<1) or (x>8 and x<14 and y>3 and y<4) or (x>8 and x<9 and y>1 and y<3) or (x>13 and x<14 and y>1 and y<3):
			res = 1.0/16.0
		else:
			res = 0.0
	return res


# velocity for the forward computation of the solution
def velocity1(x,y,t):
	return 1.0,0.0

# velocity for the forward computation of the solution
def velocity2(x,y,t):
	v1, v2 = -1.0,0.0
	if test_case == 4:
		if y>=2 and y<x-10:
			v1, v2 = 0.0, 1.0
		if y<2 and y>14-x:
			v1, v2 = 0.0,-1.0
	return v1, v2



#########################
### end of parameters ###
#########################



# discretisation of the iniale and final data
# output : une ligne par cellule : x_down_left, y_down_left, x_up_right, y_up_right
def discretisation(step_x,step_m,mu,aa,bb,cc,dd,test_case):
	vertices =  np.matrix([[]]) 
	cells =  np.matrix([[]]) 
	# step of the subdsicretisation following the mass
	#step_m = 1.0/Nm
	# initialisaton of the interval of discretisation
	x0 = aa
	x1 = aa + step_x
	init = True
	for i in range(int((bb-aa)/step_x)+1):
		x = x0
		while aa+step_x*(i+1)-x>1e-8:
			# initialisation of the intervall where is the next point
			xinf = x
			xsup = x1
			val_int = 0.0
			for j in range(1,100): # while abs(val_int-step_m)>tol_dicho
				val_int = integrate.dblquad(lambda X, Y: mu(X,Y,test_case), cc,dd, lambda t: x, lambda u: (xsup+xinf)/2)[0]
				if val_int > step_m:
					xsup = (xsup+xinf)/2
				else:
					xinf = (xsup+xinf)/2
			y0 = cc
			y1 = cc + step_x
			for j in range(int((dd-cc)/step_x)+1):
				y = y0
				while cc+step_x*(j+1)-y>1e-8:
					# initialisation of the intervall where is the next point
					yinf = y
					ysup = y1
					val_int = 0.0
					for k in range(1,100): # while abs(val_int-step_m)>tol_dicho
						val_int = integrate.dblquad(lambda X, Y: mu(X,Y,test_case), y,(ysup+yinf)/2, lambda t: x, lambda u: (xsup+xinf)/2)[0]
						if val_int > step_m**2:
							ysup = (ysup+yinf)/2
						else:
							yinf = (ysup+yinf)/2
					if x<=aa+step_x*(i+1) and xinf-x>1e-4 and y<=cc+step_x*(j+1) and yinf-y>1e-4 and val_int >1e-6:
						if init==True:
							cells = np.array([[x,y,xinf,yinf]])
							init = False
						else:
							cells = np.concatenate((cells , np.array([[x,y,xinf,yinf]])),axis=0)
					y = yinf
				y0 = y0 + step_x
				y1 = y1 + step_x
			x = xinf
		x0 = x0 + step_x
		x1 = x1 + step_x
	return cells # cordinate of the corners down-left and up-right





# compute the time for a particle to go in omega
# output : une ligne par cellule : abscisee point d'arrivee, ordonnee point d'arrivee et temps d'arrivee
def centers_cells_space_time_in_omega(velocity,centers,a,b,c,d,delta_x,dt):
	# initialisation of the final points and time
	init = True
	for i in range(len(centers)):
		zn = centers[i,0]
		wn = centers[i,1]
		tn = 0.0
		while (zn<a or zn>b or wn<c or wn>d):
			k1, l1 = velocity(zn,wn,tn)
			k2, l2 = velocity(zn+dt*k1/2,wn+dt*l1/2,tn+dt/2)
			k3, l3 = velocity(zn+dt*k2/2,wn+dt*l2/2,tn+dt/2)
			k4, l4 = velocity(zn+dt*k3,wn+dt*l3,tn+dt)
			zn = zn + (dt/6)*(k1+2*k2+2*k3+k4)
			wn = wn + (dt/6)*(l1+2*l2+2*l3+l4)
			tn = tn + dt
		if init == True:
			init = False
			space_time = np.array([[zn,wn,tn]])
		else:
			space_time = np.concatenate((space_time,np.matrix([[zn,wn,tn]])),axis=0)
	return space_time


# Cost used for the computation of the optimal permutation
# Output : real number
def cost(x0,y0,t0,x1,y1,t1):
	infinity = 10000000.0#float('inf')
	if t0<=t1:
		c = ((x0-x1)**2+(y0-y1)**2+(t0-t1)**2)**0.5
		#print(c)
		#c = abs(t0-t1)
		#c = (abs(x0-x1)+abs(y0-y1)+abs(t0-t1))
	else:
		c = infinity
	return c

# computation of the optimal permutation for hte association of the masses
# output : matrix of permutation
def permutation(centers_cells_space_time_in_omega0,centers_cells_space_time_in_omega1):
	nb_elt = len(centers_cells_space_time_in_omega0)
	# construction of the constraint matrix for the minimisation
	'''A_eq = np.zeros((2*nb_elt,nb_elt**2))
	for i in range(nb_elt):
		for j in range(nb_elt):
			A_eq[i,j+i*nb_elt] = 1.0

	for i in range(nb_elt):
		for j in range(nb_elt):
			A_eq[i+nb_elt,(j-1)*nb_elt+i] = 1.0
	b_eq = np.ones(2*nb_elt)'''

	# computation of the different cost
	costs = np.array(np.zeros((nb_elt,nb_elt)))
	for i in range(nb_elt):
		for j in range(nb_elt):
			x0 = centers_cells_space_time_in_omega0[i,0]
			y0 = centers_cells_space_time_in_omega0[i,1]
			t0 = centers_cells_space_time_in_omega0[i,2]
			x1 = centers_cells_space_time_in_omega1[j,0]
			y1 = centers_cells_space_time_in_omega1[j,1]
			t1 = centers_cells_space_time_in_omega1[j,2]
			costs[i,j] = cost(x0,y0,t0,x1,y1,t1)

	#print(np.sum(costs)/nb_elt)

	# Creates a list of all the supply nodes
	warehouses = range(nb_elt)

	# Creates a list of all demand nodes
	bars = range(nb_elt)



	# The cost data is made into a dictionary
	costs_dic = pulp.makeDict([warehouses, bars], costs,0)

	# Creates the prob variable to contain the problem data
	prob = pulp.LpProblem("Beer Distribution Problem", pulp.LpMinimize)

	# Creates a list of tuples containing all the possible routes for transport
	routes = [(w,b) for w in warehouses for b in bars]

	# A dictionary called x is created to contain quantity shipped on the routes
	x = pulp.LpVariable.dicts("route", (warehouses, bars), lowBound = 0, cat = pulp.LpInteger)

	# The objective function is added to prob first
	prob += sum([x[w][b]*costs_dic[w][b] for (w,b) in routes]), "Sum_of_Transporting_Costs"


	# Supply maximum constraints are added to prob for each supply node (warehouse)
	for w in warehouses:
		prob += sum([x[w][b] for b in bars]) == 1, "Sum_of_Products_out_of_Warehouse_%s"%w

	# Demand minimum constraints are added to prob for each demand node (bar)
	for b in bars:
		prob += sum([x[w][b] for w in warehouses]) == 1, "Sum_of_Products_into_Bar%s"%b

	# The problem data is written to an .lp file
	#prob.writeLP("BeerDistributionProblem.lp")

	# The problem is solved using PuLPs choice of Solver
	prob.solve()

	# The status of the solution is printed to the screen
	print("Status:", pulp.LpStatus[prob.status])


	# Each of the variables is printed with its resolved optimum value
	#for v in prob.variables():
	#	print v.name, "=", v.varValue

	A = np.zeros((nb_elt,nb_elt))
	for i in range(nb_elt):
		for j in range(nb_elt):
			A[i,j] = x[i][j].value()

	#print A

	# The optimised objective function value is printed to the screen
	print("Total Cost of Transportation = ", prob.objective.value())

	costs2 = A*np.transpose(np.matrix(costs))

	#print(costs)
	#print(costs2)

	'''J1 = 0
	J2 = 0
	for i in range(nb_elt):
		J1 += costs[i,i]
		J2 += costs2[i,i]

	print('J1',J1)
	print('J2',J2)'''

	return A





# follow the charatiristic of a given point
# output : 3 vectors, absisce, ordinate, temps
def runge_kutta4(x0,y0,velocity,dt,t_init,t_final):
	z_array = np.matrix([x0])
	w_array = np.matrix([y0])
	t_array = np.matrix([t_init])
	tn=t_init
	zn=x0
	wn=y0
	N = round((t_final-t_init)/dt)+1
	for k in np.arange(1,N):
		tn=t_init+k*dt
		k1, l1 = velocity(zn,wn,tn)
		k2, l2 = velocity(zn+dt*k1/2,wn+dt*l1/2,tn+dt/2)
		k3, l3 = velocity(zn+dt*k2/2,wn+dt*l2/2,tn+dt/2)
		k4, l4 = velocity(zn+dt*k3,wn+dt*l3,tn+dt)
		zn += (dt/6)*(k1+2*k2+2*k3+k4)
		wn += (dt/6)*(l1+2*l2+2*l3+l4)
		z_array = np.concatenate((z_array,np.matrix([zn])),axis=0)
		w_array = np.concatenate((w_array,np.matrix([wn])),axis=0)
		t_array = np.concatenate((t_array,np.matrix([tn])),axis=0)
	return z_array, w_array, t_array


# fiting of the final solution
def interpolate(mu_left_down_x,mu_left_down_y,mu_right_up_x,mu_right_up_y,mu_val,Ix,Iy):
	mu = np.zeros(len(Ix),len(Iy))
	#print(mu_left,mu_right,mu_val,len(mu_left))
	for i in range(len(Ix)):
		for j in range(len(Iy)):
			for k in range(np.size(mu_left)):
				if Ix[i]>=mu_left_down_x[0,k] and Ix[i]<mu_right_up_x[0,k] and Iy[i]>=mu_left_down_y[0,k] and Iy[i]<mu_right_up_y[0,k]:
					mu[i]=mu[i]+mu_val[0,k]
	return mu


def output_write():
	f = open('output.txt','w')
	f.write('discretisation of mu0')
	f.write('\n')
	f.write(str(cells0))
	f.write('\n')
	f.write('\n')
	f.write('discretization of mu1')
	f.write('\n')
	f.write(str(cells1))
	f.write('\n')
	f.write('\n')
	f.write('centers of cells of mu0')
	f.write('\n')
	f.write(str(centers0))
	f.write('\n')
	f.write('\n')
	f.write('centers of cells of mu1')
	f.write('\n')
	f.write(str(centers1))
	f.write('\n')
	f.write('\n')
	f.write('centers_cells_space_time_in_omega0')
	f.write('\n')
	f.write(str(centers_cells_space_time_in_omega0))
	f.write('\n')
	f.write('\n')
	f.write('centers_cells_space_time_in_omega1')
	f.write('\n')
	f.write(str(centers_cells_space_time_in_omega1))
	f.write('\n')
	f.write('\n')
	f.write('permutation')
	f.write('\n')
	f.write(str(permut))
	f.write('\n')
	f.write('\n')
	f.write('cells0')
	f.write('\n')
	f.write(str(cells0))
	f.write('\n')
	f.write('\n')
	f.write('cells1')
	f.write('\n')
	f.write(str(cells1))
	f.write('\n')
	f.write('\n')
	#f.write('mu_up_right_t')
	#for i in range(len(mu_up_right_t)):
	#	f.write('\n')
	#	f.write(str(mu_up_right_t[i,0]))
	return


#####################################
### beginning of the computations ###
#####################################

# discretisation of mu0 and mu : one line per cell, x
cells0 = discretisation(delta_x,step_m,mu_0,a0,b0,c0,d0,test_case)
cells1 = discretisation(delta_x,step_m,mu_1,a1,b1,c1,d1,test_case)



#print(vertices1)

print('Discretisation : OK')

print('Number of cells :', len(cells0))

# resize of vertices0 and vertices1
S = min(len(cells0),len(cells1))
cells0 = cells0[:S,:]
cells1 = cells1[:S,:]


# Compute the centers of the cells
centers0 = np.zeros((len(cells0),2))
centers1 = np.zeros((len(cells0),2))
for i in range(len(cells0)):
	centers0[i,0] = 0.5*(cells0[i,0]+cells0[i,2])
	centers0[i,1] = 0.5*(cells0[i,1]+cells0[i,3])
	centers1[i,0] = 0.5*(cells1[i,0]+cells1[i,2])
	centers1[i,1] = 0.5*(cells1[i,1]+cells1[i,3])



# computation of the position of the flow in omega
# and the minimal time to be in omega
# column 0 : vertice abscissa
# column 1 : vertice ordinate
# column 2 : abscissa in omega
# column 3 : ordinate in omega
# column 4 : time in omega
centers_cells_space_time_in_omega0 = centers_cells_space_time_in_omega(velocity1,centers0,a,b,c,d,delta_x,dt)
centers_cells_space_time_in_omega1 = centers_cells_space_time_in_omega(velocity2,centers1,a,b,c,d,delta_x,dt)




time_in_omega0=centers_cells_space_time_in_omega0[:,2]
time_in_omega1=centers_cells_space_time_in_omega1[:,2]





print('Space time in omega : OK')
time_in_omega0_incr = np.sort(time_in_omega0,axis=0) # increasing order
time_in_omega1_decr = np.sort(time_in_omega1,axis=0)[::-1,:] # decreasing order

#print(time_in_omega0_incr)
#print(time_in_omega1_decr)

#print(vertices_space_time_in_omega0)
#print(vertices_space_time_in_omega1)

#time_omega0 = -dt*floor(-time_omega0/dt);% multiple of dt superior
#time_omega1 = -dt*floor(-time_omega1/dt);% multiple of dt superior


# computation of the minimal time (mutiple of dt)
T_min = max(time_in_omega0_incr+time_in_omega1_decr)



print('Minimal time : ',T_min[0,0])

# time of controllability (multiple of dt)
T = 8.1#round((T_min + epsilon)/dt)*dt

print('Time of control',T)

# we add the time at which the cell is in omega
cells0 = np.concatenate((cells0,centers_cells_space_time_in_omega0[:,2]),axis=1)
cells1 = np.concatenate((cells1,centers_cells_space_time_in_omega1[:,2]),axis=1)

# retressissement
cells1_0_temp = cells1[i,0] + 0.5*coeff_retr_final*(cells1[i,2] - cells1[i,0])
cells1_2_temp = cells1[i,2] + 0.5*coeff_retr_final*(cells1[i,0] - cells1[i,2])
cells1_1_temp = cells1[i,1] + 0.5*coeff_retr_final*(cells1[i,3] - cells1[i,1])
cells1_3_temp = cells1[i,3] + 0.5*coeff_retr_final*(cells1[i,1] - cells1[i,3])	
cells1[i,0] = cells1_0_temp
cells1[i,2] = cells1_2_temp
cells1[i,1] = cells1_1_temp
cells1[i,3] = cells1_3_temp

for i in range(len(cells0)):
	centers_cells_space_time_in_omega1[i,2] =  T_min + epsilon/2 - centers_cells_space_time_in_omega1[i,2]

#print(len(cells0))


#print(vertices_space_time_in_omega1)

# construction of the permutation matrix of the mass
permut = permutation(centers_cells_space_time_in_omega0,centers_cells_space_time_in_omega1)

print('Permutation matrix : ok')



#c0 = centers_cells_space_time_in_omega0
#c1 = centers_cells_space_time_in_omega1
#print(cost(c0[1,0],c0[1,1],c0[1,2],c1[1,0],c1[1,1],c1[1,2])+cost(c0[7,0],c0[7,1],c0[7,2],c1[7,0],c1[7,1],c1[7,2]))
#print(cost(c0[1,0],c0[1,1],c0[1,2],c1[7,0],c1[7,1],c1[7,2])+cost(c0[1,0],c0[1,1],c0[1,2],c1[7,0],c1[7,1],c1[7,2]))
#cells1 = np.transpose(permut)*cells1



cells1 = np.matrix(permut)*np.matrix(cells1)

'''nb_elt = len(cells1)

costs = np.zeros((nb_elt,nb_elt))
for i in range(nb_elt):
	for j in range(nb_elt):
		x0 = centers_cells_space_time_in_omega0[i,0]
		y0 = centers_cells_space_time_in_omega0[i,1]
		t0 = centers_cells_space_time_in_omega0[i,2]
		x1 = centers_cells_space_time_in_omega1[j,0]
		y1 = centers_cells_space_time_in_omega1[j,1]
		t1 = centers_cells_space_time_in_omega1[j,2]
		costs[i,j] = cost(x0,y0,t0,x1,y1,t1)

#costs2 = np.matrix(permut)*np.transpose(np.matrix(costs))

#print(costs2[0,0],costs2[1,1],costs2[2,2])

J1 = np.sum(np.multiply(np.identity(nb_elt),costs))
J2 = np.sum(np.multiply(permut,costs))


print('J1',J1)
print('J2',J2)
'''

# we compute the flow associated to the boundary of the cell
# and we take the midle of the cell

#print(cells0)


for i in range(len(cells0)):
	t0 = cells0[i,4]
	t1 = cells1[i,4]



	##
	mu0_down_left_x, mu0_down_left_y, mu0_down_left_t = runge_kutta4(cells0[i,0],cells0[i,1],velocity1,dt,0.0,t0)
	mu0_up_right_x, mu0_up_right_y, mu0_up_right_t = runge_kutta4(cells0[i,2],cells0[i,3],velocity1,dt,0.0,t0)
	mu1_down_left_x, mu1_down_left_y, mu1_down_left_t = runge_kutta4(cells1[i,0],cells1[i,1],velocity2,dt,0.0,t1)
	mu1_up_right_x, mu1_up_right_y, mu1_up_right_t = runge_kutta4(cells1[i,2],cells1[i,3],velocity2,dt,0.0,t1)
	##
	mu1_down_left_x = mu1_down_left_x[::-1,:]
	mu1_down_left_y = mu1_down_left_y[::-1,:]
	mu1_down_left_t = (T-mu1_down_left_t)[::-1,:]
	mu1_up_right_x = mu1_up_right_x[::-1,:]
	mu1_up_right_y = mu1_up_right_y[::-1,:]
	mu1_up_right_t = (T-mu1_up_right_t)[::-1,:]
	#print(t0,t1,T-t1-t0)
	N01 = int(round((T-t1-t0)/dt)) # number of step between t0 and t1
	mu_down_left_x = np.matrix(np.zeros((N01-1,1)))
	mu_down_left_y = np.matrix(np.zeros((N01-1,1)))
	mu_down_left_t = np.matrix(np.zeros((N01-1,1)))
	mu_up_right_x = np.matrix(np.zeros((N01-1,1)))
	mu_up_right_y = np.matrix(np.zeros((N01-1,1)))
	mu_up_right_t = np.matrix(np.zeros((N01-1,1)))
	for j in range(N01-1):
		mu_down_left_x[j,0] = mu0_down_left_x[-1,0]+(j+1)*(mu1_down_left_x[0,0]-mu0_down_left_x[-1,0])/N01
		mu_down_left_y[j,0] = mu0_down_left_y[-1,0]+(j+1)*(mu1_down_left_y[0,0]-mu0_down_left_y[-1,0])/N01
		mu_down_left_t[j,0] = t0 + (j+1)*dt
		mu_up_right_x[j,0] = mu0_up_right_x[-1,0]+(j+1)*(mu1_up_right_x[0,0]-mu0_up_right_x[-1,0])/N01
		mu_up_right_y[j,0] = mu0_up_right_y[-1,0]+(j+1)*(mu1_up_right_y[0,0]-mu0_up_right_y[-1,0])/N01
		mu_up_right_t[j,0] = t0 + (j+1)*dt
	#if mod(mu0_h(i),delta_x)==0:
	#	mu = sol(end-length(mu(:,1))+1:end,:)
	#else:

	#retrecicement
	mu_down_left_x_temp = mu_down_left_x + 0.5*coeff_retr_x*(mu_up_right_x - mu_down_left_x)
	mu_up_right_x_temp = mu_up_right_x + 0.5*coeff_retr_x*(mu_down_left_x - mu_up_right_x)
	mu_down_left_y_temp = mu_down_left_y + 0.5*coeff_retr_y*(mu_up_right_y - mu_down_left_y)
	mu_up_right_y_temp = mu_up_right_y + 0.5*coeff_retr_y*(mu_down_left_y - mu_up_right_y)
	mu_down_left_x = mu_down_left_x_temp
	mu_up_right_x = mu_up_right_x_temp
	mu_down_left_y = mu_down_left_y_temp
	mu_up_right_y = mu_up_right_y_temp


	'''mu1_down_left_x = mu1_down_left_x + 0.5*coeff_retr_final*(mu1_up_right_x - mu1_down_left_x)
	mu1_up_right_x = mu1_up_right_x + 0.5*coeff_retr_final*(mu1_down_left_x - mu1_up_right_x)
	mu1_down_left_y = mu1_down_left_y + 0.5*coeff_retr_final*(mu1_up_right_y - mu1_down_left_y)
	mu1_up_right_y = mu1_up_right_y + 0.5*coeff_retr_final*(mu1_down_left_y - mu1_up_right_y)'''

	mu_down_left_x = np.concatenate((mu0_down_left_x,mu_down_left_x,mu1_down_left_x),axis=0)
	mu_down_left_y = np.concatenate((mu0_down_left_y,mu_down_left_y,mu1_down_left_y),axis=0)
	mu_down_left_t = np.concatenate((mu0_down_left_t,mu_down_left_t,mu1_down_left_t),axis=0)
	mu_up_right_x = np.concatenate((mu0_up_right_x,mu_up_right_x,mu1_up_right_x),axis=0)
	mu_up_right_y = np.concatenate((mu0_up_right_y,mu_up_right_y,mu1_up_right_y),axis=0)
	mu_up_right_t = np.concatenate((mu0_up_right_t,mu_up_right_t,mu1_up_right_t),axis=0)
	if i==0:
		sol_down_left_x = mu_down_left_x
		sol_down_left_y = mu_down_left_y
		sol_down_left_t = mu_down_left_t
		sol_up_right_x = mu_up_right_x
		sol_up_right_y = mu_up_right_y
		sol_up_right_t = mu_up_right_t
	else:
		sol_down_left_x = np.concatenate((sol_down_left_x,mu_down_left_x),axis=1)
		sol_down_left_y = np.concatenate((sol_down_left_y,mu_down_left_y),axis=1)
		sol_down_left_t = np.concatenate((sol_down_left_t,mu_down_left_t),axis=1)
		sol_up_right_x = np.concatenate((sol_up_right_x,mu_up_right_x),axis=1)
		sol_up_right_y = np.concatenate((sol_up_right_y,mu_up_right_y),axis=1)
		sol_up_right_t = np.concatenate((sol_up_right_t,mu_up_right_t),axis=1)


#print(len(sol_up_right_t))
#for i in range(len(sol_up_right_t)-1):
#	if sol_up_right_t[i+1,0]-sol_up_right_t[i,0]<0.1 or sol_up_right_t[i+1,0]-sol_up_right_t[i,0]>0.1:
#		print(sol_up_right_t[i+1,0]-sol_up_right_t[i,0])


#print(sol_up_right_t)

mu_val = step_m**2/np.multiply(sol_up_right_x-sol_down_left_x,sol_up_right_y-sol_down_left_y)

if output == True:
	output_write()


print("Computation of the solution : OK")


dx = np.max(sol_up_right_x-sol_down_left_x)/25
dy = np.max(sol_up_right_y-sol_down_left_y)/25
minx = np.min((sol_up_right_x,sol_down_left_x))
maxx = np.max((sol_up_right_x,sol_down_left_x))
miny = np.min((sol_up_right_y,sol_down_left_y))
maxy = np.max((sol_up_right_y,sol_down_left_y))
mint = 0.0
#maxt = float(T)

#print(maxx)


Ix = np.linspace(minx,maxx,num=np.floor((maxx-minx)/dx)+1)
Iy = np.linspace(miny,maxy,num=np.floor((maxy-miny)/dy)+1)
It = sol_down_left_t #np.linspace(mint,maxt,num=np.floor((maxt-mint)/dt)+1)



solution = np.zeros((len(It),len(Iy),len(Ix)))
for ind_t in range(len(It)):
	for ind_c in range(len(cells0)):
		ind_minx = int(len(Ix)*(sol_down_left_x[ind_t,ind_c]-minx)/(maxx-minx))
		ind_maxx = int(len(Ix)*(sol_up_right_x[ind_t,ind_c]-minx)/(maxx-minx))
		ind_miny = int(len(Iy)*(sol_down_left_y[ind_t,ind_c]-miny)/(maxy-miny))
		ind_maxy = int(len(Iy)*(sol_up_right_y[ind_t,ind_c]-miny)/(maxy-miny))
		#print(ind_minx,ind_maxx,ind_minx,ind_maxy)
		for ind_x in range(ind_minx,ind_maxx):
			for ind_y in range(ind_miny,ind_maxy):
				solution[ind_t,ind_y,ind_x]=solution[ind_t,ind_y,ind_x]+mu_val[ind_t,ind_c]


print('max sol :',np.max(solution))


solution = solution**0.25






fig = plt.figure(figsize=(7,2))

ax = plt.gca()
ax.set_aspect(1)

patch = patches.Rectangle((a,c),b-a,d-c,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(patch)
ax = plt.axes(xlim=(minx,maxx), ylim=(miny,maxy))

#print(np.max(solution))

# initialization function: plot the background of each frame
def init():
    im.set_data([[]])
    return im,
ind_t = 0

max_sol = np.max(solution)
min_sol = np.min(solution)



data = solution[ind_t,:,:]

ax.axis([minx-0.5,maxx+0.5, miny-0.5,maxy+0.5])

im = plt.imshow(solution[ind_t,:,:], animated=True, extent=[minx,maxx,miny,maxy],vmax=max_sol, vmin=min_sol,cmap="Blues",zorder=2)




def updatefig(ind_t):
	global solution
	im.set_array(solution[ind_t,:,:])
	#ax.add_patch(patch)
	return im,

anim = animat.FuncAnimation(fig, updatefig, interval=10,frames=len(It), blit=True)
anim.save('basic_animation_2D.mp4', fps=20, extra_args=['-vcodec', 'libx264'],dpi=1000)
plt.show()


for i in range(number_frame+1):
	ind_t = int(np.round((len(It)-1)*i/number_frame))

	fig = plt.figure(figsize=(7,2))
	ax = fig.add_subplot(111)
	patch = patches.Rectangle((a,c),b-a,d-c,linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(patch)
	ax.imshow(solution[ind_t,:,:], extent=[minx,maxx,miny,maxy],vmax=max_sol, vmin=min_sol,cmap="Blues")
	ax.legend(loc='upper left', shadow=True)
	ax.axis([minx-0.5,maxx+0.5, miny-0.5,maxy+0.5])
	ax.set_title('t = {0}s'.format(str(np.around(dt*ind_t,decimals=2))))
	plt.savefig('sol2D{0}'.format(str(i)))
	#plt.show()



