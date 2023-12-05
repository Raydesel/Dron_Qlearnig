#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:07:41 2022

@author: Raydesel
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import math

def normalize_angle(angle):
  while angle < -180:
    angle += 360
  while angle > 180:
    angle -= 360
  return angle


def sign(x): 
    return int(x>0)

def colum_yaw(x_y, extend_route=1):
    #Obtiene angulo yaw para los n waypoints
    yaw = []
    Ixy = []        
    incremento_x_and_y = [] #Solo ordena los datos en un arrglo de una dimension
    for y, ele in enumerate(x_y):
        for x, sub in enumerate(ele):
            incremento_x_and_y.append(sub)
            
    for i in range(len(incremento_x_and_y)-2):
        Ixy.append((incremento_x_and_y[i+2]-incremento_x_and_y[i]))
        
    for i in range(0,len(Ixy),2):

        #1er cuadrante cartesiano
        if sign(Ixy[i]) == 1 and sign(Ixy[i+1]) == 1:
            if Ixy[i] == 0:
                if Ixy[i+1] > 0:
                    yaw.append(90)
                else:
                    yaw.append(-90)#-90
            else:
                yaw.append((np.degrees(np.arctan(np.absolute(Ixy[i+1])/np.absolute(Ixy[i])))))
        #2do cuadrante cartesiano
        if sign(Ixy[i]) == 0 and sign(Ixy[i+1]) == 1:
            if Ixy[i] == 0:
                if Ixy[i+1] > 0:
                    yaw.append(90)
                else:
                    yaw.append(-90)#-90
            else:
                yaw.append((180 - np.degrees(np.arctan(np.absolute(Ixy[i+1])/np.absolute(Ixy[i])))))
        #3er cuadrante cartesiano
        if sign(Ixy[i]) == 0 and sign(Ixy[i+1]) == 0:
            if Ixy[i] == 0:
                if Ixy[i+1] > 0:
                    yaw.append(90)
                else:
                    yaw.append(-90)#-90
            else:
                yaw.append((180 + np.degrees(np.arctan(np.absolute(Ixy[i+1])/np.absolute(Ixy[i]))))) 
        #4to cuadrante cartesiano
        if sign(Ixy[i]) == 1 and sign(Ixy[i+1]) == 0:
            if Ixy[i] == 0:
                if Ixy[i+1] > 0:
                    yaw.append(90)
                else:
                    yaw.append(-90)#-90
            else:
                yaw.append((-np.degrees(np.arctan(np.absolute(Ixy[i+1])/np.absolute(Ixy[i])))))  

    #este loop es para agregar incrementos de yaw suaves al inicio de la ruta
    #tambien hay que agregar el copy(waypoint inicial) de delay,x,y,z que acompanan a este yaw
    
    for i in range(len(yaw)):
        yaw[i] = normalize_angle(yaw[i])
    
    if extend_route == 0:
        #Aqui estoy experimentando con este cambio para suavisar el giro y hacerlo en menos tiempo
        # while (not(-45 <= yaw[0] <= 45)):
        #     if (yaw[0] > 45 or yaw[0] < -45):
        #         yaw.insert(0, yaw[0]/2.0)
        while (not(-45 <= yaw[0] <= 45)):
            if yaw[0] > 45:
                yaw.insert(0, yaw[0]-45)
            if yaw[0] < -45:
                yaw.insert(0, yaw[0]+45)
    
    yaw.append(yaw[-1]) #repite el ultimo valor de yaw, porque el final no se alcanza a generar
    return yaw

def savedata(data, delay, extend_route=1):
    
    #Ordena los datos en columnas
    x_y = np.transpose(data)
    
    #Funcion que genera el vector yaw
    yaw = colum_yaw(x_y, extend_route)
    
    #aumento de x_y colums para agregar yaw de suavisado
    n_copy = len(yaw) - len(x_y)
    
    x_copy = np.repeat(x_y[0,0], n_copy).reshape(n_copy,1)
    y_copy = np.repeat(x_y[0,1], n_copy).reshape(n_copy,1)
    x_y_copy = [x_copy,y_copy]

    x_y_copy = np.transpose(x_y_copy).reshape(n_copy,2)
    
    x_y = np.insert(x_y, 0, x_y_copy, 0)
    
    #arreglo para ordenar columnas
    delay_x_y_z_yaw = np.zeros((len(x_y), 5))
    
    #ordenamiento de x y y en columnas 2 y 3
    for y, ele in enumerate(x_y.T, 1):
        for x, sub in enumerate(ele):
            delay_x_y_z_yaw[x, y] = sub
    
    #ordena yaw en columna 4
    y=4
    for x, sub in enumerate(yaw): 
        delay_x_y_z_yaw[x, y] = sub
    
    #genera columna delay
    delay_vector = np.ones(len(x_y))
    delay = delay_vector*delay
    #retardo giro 180 suave
    if n_copy > 0:
        for i in range(n_copy):
            delay[i+1] = 2.0
    
    #ordena delay en la primera columna
    y=0
    for x, sub in enumerate(delay): 
        delay_x_y_z_yaw[x, y] = sub
    
    #genera columna z
    z = 2.0
    z_vector = np.ones(len(x_y))
    z = z_vector*z
    
    #ordena z en columna tres
    y=3
    for x, sub in enumerate(z): 
        delay_x_y_z_yaw[x, y] = sub
    
    return delay_x_y_z_yaw
