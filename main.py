# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 19:04:45 2024

@author: hibado
"""
import numpy as np
import cv2 
from copy import deepcopy
import  matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class Ising:
    def __init__(self, nodes, edges,beta, J):
        self.nodes = nodes
        self.edges = edges
        self.J = J
        self.beta = beta
        self.add_neighbor()
        self.N = len(nodes)
        
    def add_neighbor(self):
        for i,j in self.edges:
            self.nodes[i].neighbors[j] = self.nodes[j]
    
    def step(self, beta):
        pass
        
    def get_state(self):
        X = np.array([x.spin for x in nodes.values()])
        return X.copy()
    
    def get_magnetization(self):
        X = self.get_state()
        M = sum(X)/self.N
        return M
    
class Electron:
    def __init__(self, id_, loc, spin):
        self.id = id_
        self.loc = loc
        self.spin = spin
        self.neighbors = {}
        
def dE(sys, electron, h):
    Ei = 0
    Ej = 0
    for neighbor in electron.neighbors.values():
        Ei += -electron.spin*neighbor.spin
        Ej += electron.spin*neighbor.spin
    return deepcopy(Ej-Ei + external*electron.spin)
 
beta = 0.01
external = 0.1
radius = 1
# stencil = np.array([[-1,0], [1,0], [0,-1], [0,1]])
stencil = np.array([[i,j] for i in np.arange(-radius, radius+1, 1) for j in np.arange(-radius, radius+1, 1)])
w, h = 100, 100
N = w*h
id_map = np.array(range(N)).reshape((h,w))

nodes={}
edges = []
for i in range(h):
    for j in range(w):
        id_ = id_map[i,j]
        nodes[id_]=Electron(id_, [i,j], np.random.choice([-1,1]))
        idx = stencil + [i,j]
        idx[:,0] = idx[:,0]%h
        idx[:,1] = idx[:,1]%w

        # idx = idx[np.where((idx[:,0]>=0) & (idx[:,1]>=0) & (idx[:,0]<h) & (idx[:,1]<w) )]
        edges +=  [[id_, id_map[k[0],k[1]]] for k in idx]

        
sys=Ising(nodes, edges, beta,1)
T = 500
flip_num = 50000
states=np.zeros((T, h,w))
beta = np.linspace(0,0.5,T)
Ms = []
for t in range(T):
    M = []
    for _ in range(flip_num):
        flip = np.random.randint(0,N)
        delta = dE(sys, nodes[flip], external)
        if delta>0:
            A = np.exp(-beta[t]*delta)
        else:
            A = 1
            
        if np.random.rand()<A:
            nodes[flip].spin *= -1   
        # M.append(sys.get_magnetization())
        
    Ms.append(sys.get_magnetization())
        
    state = sys.get_state().reshape(h,w)
    im = (state+2)//2*255
    states[t,:,:] = sys.get_state().reshape(h,w)
    im = cv2.resize(im.astype(np.uint8), (1000,1000), interpolation=0)
    cv2.imshow("ising", im)
    cv2.waitKey(3)
    
# for t in range(T):
#     im = states[t,:,:]
#     im = cv2.resize(im, (h*100,w*100), interpolation=0)
#     cv2.imshow("ising", im)
#     cv2.waitKey(3)
cv2.destroyAllWindows()
#%%
plt.plot(beta, Ms)

plt.plot(beta, ((1-(np.sinh(2*beta))**(-4))**(1/8)))

#%%

fig = plt.figure()    
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.axis([0, w, 0, h])
plt.axis('square')
plt.ylim(0, h)
plt.xlim(0 , w)
def animate(i):
    ax.clear()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    grid = ax.imshow(states[i], cmap="Greys_r")
    ax.set_title(r'$\beta = $'+str(round(beta[i],2)))
    return [grid]
        
ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=T)    
ani.save("ising.mp4", dpi=500,  writer='ffmpeg')   
