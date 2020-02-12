from __future__ import division
import numpy as np
import scipy.integrate as integrate


def runge_kutta4(x0,velocity,dt,t_init,t_final):
	y_array = np.matrix([x0])
	t_array = np.matrix([t_init])
	tn=t_init
	yn=x0
	N = round((t_final-t_init)/dt)+1

	for k in np.arange(1,N):
		tn=t_init+k*dt
		k1 = velocity(yn,tn)
		k2 = velocity(yn+dt*k1/2,tn+dt/2)
		k3 = velocity(yn+dt*k2/2,tn+dt/2)
		k4 = velocity(yn+dt*k3,tn+dt)
		yn = yn + (dt/6)*(k1+2*k2+2*k3+k4)
		y_array = np.concatenate((y_array,np.matrix([yn])),axis=0)
		t_array = np.concatenate((t_array,np.matrix([tn])),axis=0)
	return y_array, t_array


def discretisation(step_x,Nm,mu,c,d,test_case):


	# step of the subdsicretisation following the mass
	step_m = 1.0/Nm
	
	# initialisaton of the interval of discretisation
	x0 = c
	x1 = c + step_x
	
	init = True
	
	for i in range(int((d-c)/step_x)+1):
		x = x0
		#print(x)
		while c+step_x*(i+1)-x>1e-8:
			# initialisation of the intervall where is the next point
			#print(x)
			xinf = x
			xsup = x1
			val_int = 0.0
			for j in range(1,100): # while abs(val_int-step_m)>tol_dicho
				val_int = integrate.quad(lambda x: mu(x,test_case),x,(xsup+xinf)/2)[0]
				if val_int > step_m:
					xsup = (xsup+xinf)/2
				else:
					xinf = (xsup+xinf)/2
			if x<=c+step_x*(i+1) and xinf-x>1e-4 and val_int >1e-6:
				if init==True:
					mu_h =  np.matrix([(xinf+x)/2])
					mu_h_val =  np.matrix([val_int/(xinf-x)])
					mu_h_left = np.matrix([x])
					#print(mu_h_left)
					mu_h_right = np.matrix([ xinf])
					init = False
				else:
					mu_h = np.concatenate((mu_h , np.matrix([(xinf+x)/2])),axis=0)
					mu_h_val = np.concatenate((mu_h_val, np.matrix([val_int/(xinf-x)])),axis=0)
					mu_h_left = np.concatenate((mu_h_left , np.matrix([x])),axis=0)
					mu_h_right = np.concatenate((mu_h_right ,np.matrix([ xinf])),axis=0)
			x = xinf
		x0 = x0 + step_x
		x1 = x1 + step_x
	return mu_h,mu_h_val,mu_h_left,mu_h_right



