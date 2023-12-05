#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:00:28 2022

@author: Raydesel
"""
import numpy as np
import pandas as pd

def bezier(route_to_coordinate, points):
    x_y = pd.DataFrame(route_to_coordinate)
    
    x_y = np.asarray(x_y)
    
    #Especificar las coordenadas x e y
    x = x_y[:,0] 
    y = x_y[:,1]
    #Tamaño del arreglo de coordenadas
    m = len(x)
    #Tamaño del arreglo de coordenadas menos 1
    n = m - 1
    #si points = 1000 entonces 1/1000 = 0.001
    points = 1/points 
    #Vector de valores eje x para evaluar la curva 
    t = np.arange(0,1,points) #si points = 0.001 = 1/1000 = 1000 points
    #Tamaño del vector t
    h = len(t)
    #Inicializa variables
    J = np.zeros((m,h)) 
    X = np.zeros((m,h))
    Y = np.zeros((m,h))
    X1 = np.zeros(h)
    Y1 = np.zeros(h)
    #Ciclo principal
    for i in range(0,m):
        for j in range(0,h):
            ni = np.math.factorial(n)/(np.math.factorial(i)*np.math.factorial(n-(i)))
            J[i,j] = ni*t[j]**(i)*(1-t[j])**(n-(i))
            X[i,j] = J[i,j]*x[i]   
            Y[i,j] = J[i,j]*y[i] 
    
        for j in range(0,h):    
            X1[j]=np.sum(X[:,j]);
            Y1[j]=np.sum(Y[:,j]);
            
    data = [X1,Y1]
    xy = [x,y]
    return data