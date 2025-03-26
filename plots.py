# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:44:32 2025

@author: hibado
"""

import matplotlib.pyplot as plt 
import numpy as np
np.printoptions(precision=2)

x = np.linspace(-2,2, 100)

plt.figure(dpi=600)
plt.plot(x, np.min([np.ones(x.shape), np.exp(-x)], axis=0), 'k', label=r"$\min\{1,e^{-\beta \Delta E}\}$")
plt.plot(x, np.max([np.ones(x.shape), np.exp(-x)], axis=0), 'k--')
plt.xlabel(r"$\Delta E$")
plt.ylabel(r"$\mathbb{P}(X_j|X_i)$")
plt.ylim([-0.1,1.5])
plt.legend()
#%%
x = np.linspace(-2,2, 1000)
beta = 2

plt.figure(dpi=600)

y1 = np.tanh(beta*x)
plt.plot(x,y1, label=r"$\tanh(|\mathcal{N}|\beta JM)$")

y2 = x

plt.plot(x,y2, label="M")
plt.plot([-2,2], [0,0], "k--")
plt.ylim([-1.5,1.5])
plt.legend()
plt.title(r"$|\mathcal{N}|\beta J$ = " +str(beta))
#%%
from matplotlib.animation import FuncAnimation, PillowWriter
fig = plt.figure(dpi=600)    
ax = plt.gca()
beta = np.linspace(0.5,3, 200)
plt.ylim(-1.5,1.5)
def animate(t):
    y1 = np.tanh(beta[t]*x)
    ax.clear()
    ax.set_ylim(-1.5,1.5)
    ax.plot(x,y1, label=r"$\tanh(|\mathcal{N}|\beta JM)$")
    ax.plot(x,y2, label="M")
    obj = ax.plot([-2,2], [0,0], "k--")
    ax.set_title(r"$|\mathcal{N}|\beta J$ = " +str(round(beta[t], 2)))
    return obj

ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=len(beta))    
ani.save("resources/beta_plot.gif", dpi=300, writer=PillowWriter(fps=25))

#%%
