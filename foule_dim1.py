from __future__ import division
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.interpolate
from matplotlib import animation
from matplotlib.path import Path
import matplotlib.patches as patches


from permutation_time import *
from discretisation import *
from params import *
from interpolate import *

# discretisation of mu0 and mu
# mu0_h and mu0_h are the midel of the cells
mu0_h, mu0_h_val, mu0_h_left, mu0_h_right = discretisation(delta_x,Nm,mu_0,a0,b0,test_case)
mu1_h, mu1_h_val, mu1_h_left, mu1_h_right = discretisation(delta_x,Nm,mu_1,a1,b1,test_case)

#print(mu0_h_val)

# resize of mu0h and mu1h
S = min(len(mu0_h),len(mu1_h))
mu0_h = mu0_h[:S,0]
mu1_h = mu1_h[:S,0]
mu0_h_val = mu0_h_val[:S,0]
mu1_h_val = mu1_h_val[:S,0]
mu0_h_left = mu0_h_left[:S+1,0]
mu0_h_right = mu0_h_right[:S+1,0]
mu1_h_left = mu1_h_left[:S+1,0]
mu1_h_right = mu1_h_right[:S+1,0]


# computation of the position of the flow in omega
# and the minimal time to be in omega
point_omega0,time_omega0 = time_in_omega(velocity,mu0_h,a,b,delta_x,dt)
point_omega1,time_omega1 = time_in_omega(velocity2,mu1_h,a,b,delta_x,dt)

#print(time_omega1)


#time_omega0 = -dt*floor(-time_omega0/dt);% multiple of dt superior
#time_omega1 = -dt*floor(-time_omega1/dt);% multiple of dt superior


# computation of the minimal time (mutiple of dt)
T_min = minimal_time(time_omega0,time_omega1)

#print(T_min)

# time of controllability (multiple of dt)
T = 8.1#T_min + epsilon

#print(time_omega1)

# construction of the permutation matrix of the mass
time_omega1 = np.matrix(T*np.ones((len(time_omega1),1))) - time_omega1
#sigma = permutation(point_omega0, time_omega0,point_omega1, time_omega1)
#mu1_h = sigma*mu1_h
#point_omega1 = sigma*point_omega1
#time_omega1 = sigma*time_omega1





# we compute the flow associated to the boundary of the cell
# and we take the midle of the cell

#fig1 = figure;
#hold on 

#print(T)

for i in range(S):
	t0 = time_omega0[i,0]
	t1 = time_omega1[i,0]
	#print(type(t0))
	mu0_left, mu0_time = runge_kutta4(mu0_h_left[i,0],velocity,dt,0.0,t0)
	mu0_right, mu0_time = runge_kutta4(mu0_h_right[i,0],velocity,dt,0.0,t0)
	mu1_left, mu1_time = runge_kutta4(mu1_h_left[i,0],velocity2,dt,0.0,T-t1)
	mu1_right, mu1_time = runge_kutta4(mu1_h_right[i,0],velocity2,dt,0.0,T-t1)
	mu1_time = (T-mu1_time)[::-1,:]
	mu1_left = mu1_left[::-1,:]
	mu1_right = mu1_right[::-1,:]
	N01 = int(round((t1-t0)/dt)) # number of step between t0 and t1
	mu_left = np.matrix(np.zeros((N01-1,1)))
	mu_right = np.matrix(np.zeros((N01-1,1)))
	mu_time = np.matrix(np.zeros((N01-1,1)))
	for j in range(N01-1):
		mu_left[j,0] = mu0_left[-1,0]+(j+1)*(mu1_left[0,0]-mu0_left[-1,0])/N01
		mu_right[j,0] = mu0_right[-1,0]+(j+1)*(mu1_right[0,0]-mu0_right[-1,0])/N01
		mu_time[j,0] = t0 + (j+1)*dt
	#if mod(mu0_h(i),delta_x)==0:
	#	mu = sol(end-length(mu(:,1))+1:end,:)
	#else:
	mu_left = np.concatenate((mu0_left,mu_left,mu1_left),axis=0)
	mu_right = np.concatenate((mu0_right,mu_right,mu1_right),axis=0)
	mu_x = (mu_right+mu_left)/2
	mu_t = np.concatenate((mu0_time,mu_time,mu1_time),axis=0)
	#print(mu_right - mu_left)
	mu_val = mu0_h_val[i,0]*(mu0_h_right[i,0] - mu0_h_left[i,0])/(mu_right - mu_left)
	#plot3(mu_x,mu_t,mu_val)
	#size(mu_left)
	#mu(end,:)
	if i==0:
		sol_x = mu_x
		sol_left = mu_left
		sol_right = mu_right
		sol_t = mu_t
		sol_val = mu_val
	else:
		sol_x = np.concatenate((sol_x,mu_x),axis=1)
		sol_left = np.concatenate((sol_left,mu_left),axis=1)
		sol_right = np.concatenate((sol_right,mu_right),axis=1)
		sol_t = np.concatenate((sol_t,mu_t),axis=1)
		sol_val = np.concatenate((sol_val,mu_val),axis=1)


print("Computation of the solution : OK")


dx = np.max(sol_right-sol_left)/5
minx = min(np.min(sol_left),np.min(sol_right))
maxx = max(np.max(sol_left),np.max(sol_right))
mint = np.min(sol_t)
maxt = np.max(sol_t)


Ix = np.linspace(minx,maxx,num=np.floor((maxx-minx)/dx)+1)
It = np.linspace(mint,maxt,num=np.floor((maxt-mint)/dt)+1)

# computation of the maximum
#max_val = 0.0
#for i in range(Ix):
#	for j in range(


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(7,3.5))

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


Ix2 = np.linspace(minx,maxx,num=np.floor((maxx-minx)/(40*dx))+1)
#print(len(Ix),len(Ix2))

print("minimal time : ",T_min)

# animation function.  This is called sequentially
def animate(i):
	i = 40*i
	x = Ix2
	y = np.zeros(len(Ix2))
	z = interpolate(sol_left[i,:],sol_right[i,:],sol_val[i,:],Ix)
	for i in range(len(Ix2)):
		y[i] = np.average(z[40*i:40*(i+1)])
	line.set_data(x, y)
	#ax.set_title('t = {0}s'.format(str(np.around(dt*i*100,decimals=2))))
	return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
if Animation == True:
	ax = plt.axes(xlim=(minx,maxx), ylim=(0,2))
	#cmap = ListedColormap(['black'])
	line, = ax.plot([], color='black',label=r"$\mu(t)$", lw=2)
	codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]
	minscreen = 0
	maxscreen = 2
	verts = [(a,minscreen), (b,minscreen), (b,maxscreen), (a, maxscreen) , (a,minscreen) ]
	path = Path(verts, codes)
	patch = patches.PathPatch(path, facecolor='tomato', lw=1,label="Control region")
	ax.add_patch(patch)
	ax.legend(loc='upper left', shadow=True)
	ax.axis([minx,maxx, minscreen,maxscreen])
	#ax.set_xlabel('Space')
	#ax.set_ylabel(r"$\mu(t)$")
	anim = animation.FuncAnimation(fig, animate,frames=int(np.floor(len(It)/40))+1, interval=1, blit=True)
	anim.save('basic_animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
	plt.show()

# characteritics
if charac == True:
	plt.scatter(sol_x,sol_t,marker='.')
	plt.show()




if screen == True:
	minscreen = 0
	maxscreen = 2
	N=100
	for i in range(int(1+T/(N*dt))):
		fig = plt.figure(figsize=(5,2.5))
		ax = fig.add_subplot(111)
		z = interpolate(sol_left[N*i,:],sol_right[N*i,:],sol_val[N*i,:],Ix)
		y = np.zeros(len(Ix2))
		for j in range(len(Ix2)):
			y[j] = np.average(z[40*j:40*(j+1)])
		verts = [(a,minscreen), (b,minscreen), (b,maxscreen), (a, maxscreen) , (a,minscreen) ]
		codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]
		path = Path(verts, codes)
		patch = patches.PathPatch(path, facecolor='tomato', lw=1,label="Control region")
		ax.add_patch(patch)
		ax.plot(Ix2,y,label=r"$\mu(t)$",color='black')
		#plt.plot([a+0.1,b-0.1],[-0.005,-0.005],linewidth=5,color='r',label="Control region")
		ax.legend(loc='upper left', shadow=True)
		ax.axis([minx,maxx, minscreen,maxscreen])
		ax.set_xlabel('Space')
		#ax.set_ylabel(r"$\mu(t)$")
		ax.set_title('t = {0}s'.format(str(np.around(dt*i*100,decimals=2))))
		plt.savefig('sol{0}'.format(str(i)))
		plt.close()
	fig = plt.figure(figsize=(5,2.5))
	ax = fig.add_subplot(111)
	z = interpolate(sol_left[-1,:],sol_right[-1,:],sol_val[-1,:],Ix)
	y = np.zeros(len(Ix2))
	for j in range(len(Ix2)):
		y[j] = np.average(z[40*j:40*(j+1)])
	verts = [(a,minscreen), (b,minscreen), (b,maxscreen), (a, maxscreen) , (a,minscreen) ]
	codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]
	path = Path(verts, codes)
	patch = patches.PathPatch(path, facecolor='tomato', lw=1,label="Control region")
	patch = patches.PathPatch(path, facecolor='tomato', lw=1,label="Control region")
	ax.add_patch(patch)
	ax.plot(Ix2,y,label=r"$\mu(t)$",color='black')
	#plt.plot([a+0.1,b-0.1],[-0.005,-0.005],linewidth=5,color='r',label="Control region")
	ax.legend(loc='upper left', shadow=True)
	ax.axis([minx,maxx, minscreen,maxscreen])
	ax.set_xlabel('Space')
	#ax.set_ylabel(r"$\mu(t)$")
	ax.set_title('t = {0}s'.format(str(T)))
	plt.savefig('sol_final')
	plt.close()




#for i in range(np.size(sol_t[:,0])):
#	mu = interpolate(sol_left[i,:],sol_right[i,:],sol_val[i,:],Ix)
#	plt.plot(Ix,mu)

#print(np.size(Ix),np.size(mu))



#plt.show()

#print(Ix,It)
#print(sol_t)

#print(sol_x,sol_t,sol_val)
# save solution 
#X = sol_x
#Time = sol_t
#Z = sol_val


#print(minx,maxx)


#It = mu_t
#xq,yq = np.meshgrid(Ix,It)

# Interpolate; there's also method='cubic' for 2-D data such as here
#zi = scipy.interpolate.griddata((X, Time), Z, (xq, yq), method='linear')

#plt.imshow(zi, vmin=Z.min(), vmax=Z.max(), origin='lower',
#           extent=[minx,maxx,mint,maxt])
#plt.colorbar()




#vq = np.griddata(X,Time,Z,xq,yq,'nearest')
#vq = gridfit(X,Time,Z,xq,yq);
#fig3 = figure;
#mesh(xq,yq,vq);


#imagesc(Ix,It,vq);
#xlabel('space'); 
#ylabel('time'); 
#colorbar

#print(max(Z))

