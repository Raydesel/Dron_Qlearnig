#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:14:50 2022

@author: Raydesel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def bspline_with_order(order, points, route_to_coordinate):
    #Especificar las coordenadas x e y 
    x_y = pd.DataFrame(route_to_coordinate)
    x_y = np.asarray(x_y)
    x = x_y[:,0]
    y = x_y[:,1]
    #tamaño del arreglo de coordenadas
    n = len(x)
    #ciclo principal
    p = np.empty((0,2))
    
    for i in range(0,n):
        #coordenadas de la trayectoria
        p = np.append(p,np.matrix([x[i],y[i]]),axis=0)
        if (i+1 >= order): 
            T = np.linspace(0,1,i+1-order+2)
            yr = np.linspace(0,1,points+1) #modificar numero de puntos,se puede expandir a 999 o mas, pero por default se dejara en 101 como los otros metodos usados en q_learning_dron_v5
            m = len(p)
            nn = len(yr)
            X = np.zeros((order,order))
            Y = np.zeros((order,order))
            a = T[0]
            b = T[-1]
            Tr = np.concatenate([np.ones(order-1)*a,T,np.ones(order-1)*b],axis=0);
            p_spl = np.empty((0,2))
            for l in range(0,nn):
                t0 = yr[l]
                h = 1*(t0 >= Tr)
                idd = np.array(np.nonzero(h))
                k = idd[0,-1]
                if (k+1 > m):
                    i
                    break
                    
                xx = p[k-order+1:k+1,0]
                yy = p[k-order+1:k+1,1]
    
                X[:,0] = np.reshape(xx,(1,order))
                Y[:,0] = np.reshape(yy,(1,order))
                
                for ii in range(1,order):
                    for j in range(ii,order):
                        num = t0-Tr[k-order+j+1]
                        if num == 0:
                            weight = 0
                        else:
                            s = Tr[k+j-ii+1]-Tr[k-order+j+1]
                            weight = num/s
    
                        X[j,ii] = (1-weight)*X[j-1,ii-1] + weight*X[j,ii-1]
                        Y[j,ii] = (1-weight)*Y[j-1,ii-1] + weight*Y[j,ii-1]
                    
                p_spl = np.append(p_spl,np.matrix([X[order-1,order-1],Y[order-1,order-1]]),axis=0)
    
    #ordenamiento de columnas y reshape, para dar como entrada a savedata()
    columna_x=np.array(p_spl[:,0])
    shape_columna_x=columna_x.shape[0]
    columna_x=columna_x.reshape(shape_columna_x,)
    
    columna_y=np.array(p_spl[:,1])
    columna_y=columna_y.reshape(shape_columna_x,)#misma dimension que columna x
    
    bspline_list_columsxy = [columna_x,columna_y]
    return bspline_list_columsxy

#Coordenadas en x
#x = [10,10,9,8,7,6,5,4,3,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,0,-1,-2,-3,-4,-5,-6,-7,-8]
#Coordenadas en y
#y = [-8,-7,-7,-7,-7,-7,-7,-7,-7,-7,-6,-5,-4,-3,-3,-2,-1,0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,8,8]
# route_to_coordinate = [(10, -8), (9, -8), (8, -8), (7, -8), (6, -8), (5, -8), (5, -7), (4, -7), (3, -7), (2, -7), (2, -6), (2, -5), (1, -5), (1, -4), (1, -3), (1, -2), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-2, 4), (-2, 5), (-3, 5), (-3, 6), (-4, 6), (-5, 6), (-6, 6), (-6, 7), (-7, 7), (-7, 8), (-8, 8)]
# #Orden del betaspline
# order = 3
# spl = bspline_with_order(order, route_to_coordinate)
# spl = np.transpose(bspline_with_order(order, route_to_coordinate))
# spl_2 = np.transpose(bspline_with_order(order, route_to_coordinate))
# spl=np.concatenate((spl,spl_2),axis=0)
# plt.plot(spl[0],spl[1],'k')
# plt.plot(p_spl[:,0],p_spl[:,1])
# plt.plot(x,y,'*r')    
