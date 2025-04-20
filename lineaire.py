import numpy as np
import numpy.linalg as lg
import scipy.linalg as slg
from time import time

A=np.array([1,1],[0,2])
B=np.array([1,0],[1,2])
C=np.array([1,1],[0,1])
D=np.array([1,1],[1,0])
E=np.array([1,2],[2,4])
F=np.array([0,1,0],[1,0,1],[0,1,0])
G=np.array([11,-5,5],[-5,3,-3],[5,-3,3])

m = [A,B,C,D,E,F,G]


for i in m:
vA = lg.eig(A)
vB = lp.eig(B)
vC = lp.eig(C)
vD = lp.eig(D)
vE = lp.eig(E)
vF = lp.eig(F)
vG = lp.eig(G)

diagA = np.array([vA[0], vB])
