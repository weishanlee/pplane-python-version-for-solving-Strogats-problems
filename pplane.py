# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 2023

@author: Wei-shan

python codes mimicking pplane to plot phase portrait for solving Strogatz's problems.

Reference:
https://scicomp.stackexchange.com/questions/40239/is-there-a-python-version-of-the-ode-tool-pplane
https://scipy-user.scipy.narkive.com/RU19ShQ4/nullclines-for-nonlinear-odes
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
https://stackoverflow.com/questions/43150872/number-of-arrowheads-on-matplotlib-streamplot
Strogatz, Nonlinear Dynamics and Chaos, 2nd Ed.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import fsolve
from numpy import linalg as LA
from matplotlib.markers import MarkerStyle
import sys

# Grid of x, y points
nx, ny = 50, 50
minX, maxX = -4, 3
minY, maxY = -4, 3

def eqnXDotYDot(x, y): 
    """
    Modify equations here
    """
    dx = x+np.exp(-y)
    dy = -y 
    return  ( dx, dy )

def func(variables):
    x, y = variables
    dx, dy = eqnXDotYDot(x, y)
    return (dx, dy)
    
def fixedPoints(X, Y):
    fPt = ()
    for i in range(nx):
        for j in range(ny):
            xFixedPoint, yFixedPoint = fsolve(func,(X[i,j],Y[i,j]))
            xFixedPoint = round(xFixedPoint,3)
            yFixedPoint = round(yFixedPoint,3)
            if all(np.isclose(eqnXDotYDot(xFixedPoint, yFixedPoint), [0.0, 0.0])) == True:
               fPt += ((xFixedPoint, yFixedPoint),)
    fPt = tuple( set(fPt) )
    return fPt


def jacobian(fs, xs, h=1e-4):
    """
    Reference: Gezerlis, Numerical Methods in Phyisics with Python, p.284
    """
    n = np.asarray(xs).size
    iden = np.identity(n)
    Jf = np.zeros((n,n))
    fs0 = fs(xs)
    for j in range(n):  # through columns to allow for vector addition
        fs1 = fs(xs+iden[:,j]*h)
        Jf[:,j] = ( np.asarray(fs1) - np.asarray(fs0) )/h
    return Jf

# find stability of a single fixed point
def sFPt(fPt):
    tolerance = 1e-8
    fPtX = fPt[0]
    fPtY = fPt[1]
    w= LA.eigvals( jacobian(func,(fPtX,fPtY)) )
    tau = w[0] + w[1]
    Delta = w[0] * w[1]
    if abs(tau)<tolerance and Delta>0:
       print("({:.3f},{:.3f}): center, linearization failed.".format(fPtX,fPtY)) 
       typeStability = "BC" # borderline case       
    elif abs(Delta)<tolerance:
       print("({:.3f},{:.3f}): line of fixed points, linearization failed.".format(fPtX,fPtY))  
       typeStability = "BC" # borderline case        
    elif tau>0 and Delta>0 and (tau**2-4*Delta)<tolerance:
       if w[0]!=w[1]:
          print("({:.3f},{:.3f}): unstable star nodes.".format(fPtX,fPtY))
          typeStability = "unstable"  
       if abs(w[0]-w[1])<tolerance:
          print("({:.3f},{:.3f}): unstable degenerate nodes.".format(fPtX,fPtY))
          typeStability = "unstable" 
    elif tau<0 and Delta>0 and (tau**2-4*Delta)<tolerance:
       if w[0]!=w[1]:
          print("({:.3f},{:.3f}): stable star nodes.".format(fPtX,fPtY))
          typeStability = "stable"  
       if abs(w[0]-w[1])<tolerance:
          print("({:.3f},{:.3f}): stable degenerate nodes.".format(fPtX,fPtY))     
          typeStability = "stable"
    elif tau>0 and Delta>0 and (tau**2-4*Delta>0):
        print("({:.3f},{:.3f}): unstable nodes".format(fPtX,fPtY))
        typeStability = "unstable"
    elif tau>0 and Delta>0 and (tau**2-4*Delta<0):
        print("({:.3f},{:.3f}): unstable spiral".format(fPtX,fPtY))
        typeStability = "unstable" 
    elif tau<0 and Delta>0 and (tau**2-4*Delta>0):
        print("({:.3f},{:.3f}): stable nodes".format(fPtX,fPtY))
        typeStability = "stable"
    elif tau<0 and Delta>0 and (tau**2-4*Delta<0):
        print("({:.3f},{:.3f}): stable spiral".format(fPtX,fPtY))
        typeStability = "stable"                
    elif Delta<0:
        print("({:.3f},{:.3f}): saddle".format(fPtX,fPtY))
        typeStability = "saddle"
    else:
        print("({:.3f},{:.3f}): other cases. Check line 112".format(fPtX,fPtY))
        typeStability = "None"
        sys.exit()
    return typeStability
    
x = np.linspace(minX, maxX, nx)
y = np.linspace(minY, maxY, ny)
X, Y = np.meshgrid(x, y)    

# field vector
dx, dy = eqnXDotYDot(X,Y)
# plot phase portrait with vector field
plt.figure()
plt.title("Phase Portrait")
ax = plt.gca()
#fig, ax = plt.subplots()
plt.minorticks_on()
minorLocatorX = AutoMinorLocator(2) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis

speed = 5*np.sqrt(dx**2 + dy**2) # the coefficient may need to modify for 
                                 # different cases
lw = speed  / speed.max()

ax.streamplot(x, y, dx, dy, linewidth=lw, density=2,color='b', arrowstyle='-')#,
              #broken_streamlines=False) # valid for matplolib version  > 3.6.0
             
ax.contour(X,Y,dx,levels=[0], linewidths=1, colors='r')
ax.contour(X,Y,dy,levels=[0], linewidths=1, colors='y')

fPt = fixedPoints(X, Y)
#%%
for i in range(len(fPt)):
    typeStability = sFPt(fPt[i])
    if typeStability == "stable":
        ax.scatter(fPt[i][0],fPt[i][1],s = 80, facecolors='k',edgecolors='k')
    elif typeStability == "unstable":
        ax.scatter(fPt[i][0],fPt[i][1],s = 80, facecolors='none',edgecolors='k')
    elif typeStability == "saddle":    
        ax.scatter(fPt[i][0],fPt[i][1],s = 80, marker=MarkerStyle("o", fillstyle="right"),facecolors='k',edgecolors='k')
        ax.scatter(fPt[i][0],fPt[i][1],s = 80, marker=MarkerStyle("o", fillstyle="left"),facecolors='none',edgecolors='k')
    elif typeStability == "BC":
        ax.scatter(fPt[i][0],fPt[i][1],s = 80, marker=r'$?$',facecolors='none',edgecolors='k')
    else:
        print("({:.3f},{:.3f}): other cases. Check line 158".format(fPt[i][0],fPt[i][1]))

# Grid for placing quivers
nx, ny = 10, 10
x = np.linspace(minX, maxX, nx)
y = np.linspace(minY, maxY, ny)
X, Y = np.meshgrid(x, y)  
dx, dy = eqnXDotYDot(X,Y) 
q = ax.quiver(X, Y, dx, dy,color='g',angles = 'uv', headlength=2,headaxislength=2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim(1.05*minX, 1.05*maxX)
ax.set_ylim(1.05*minY, 1.05*maxY)
ax.set_aspect('equal')
plt.grid(True)
plt.savefig("pplane.png")
plt.tight_layout()
plt.show()