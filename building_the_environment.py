#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:58:07 2022

@author: Raydesel
"""
import cv2 
import numpy as np
import sklearn.feature_extraction as skf

#input image20x20pxl
def building_the_environment(image, resolution, diagonals, custom_reward, densidad):
    
    ################### Cargar binarymap.png a matriz numpy

    #use 0 to read image in grayscale mode
    image = cv2.imread(image, 0) 
    
    #normalizacion de la imagen para tener solo 1 y 0
    image = image/255
    
    #Esto reemplazará todos los valores mayores o iguales a 0.5 con 1
    image = np.where(image >= 0.5, 1, 0)
    
    ################### Aplicar logica de muros a matriz image 
    
    # Padding replicate top y left
    imgpad = cv2.copyMakeBorder(image, top=1, bottom=0, left=1, right=0, borderType=cv2.BORDER_REPLICATE)
    
    #Ciclo para aplicar logica de muros
    for y, ele in enumerate(image):
        for x, sub in enumerate(ele): 
            if image[y, x] == 0: 
                imgpad[y, x] = 0        #actual
                imgpad[y, x+1] = 0      #enfrente 
                imgpad[y+1, x] = 0      #abajo
                imgpad[y+1, x+1] = 0    #esquina
    
    ################# Construir matriz de estados
    
    edos = np.arange(1, ((resolution+1)*(resolution+1)+1)).reshape(resolution+1,resolution+1)
    
    #estados validos
    imgpad_edos = imgpad*edos
       
    #mapeo location to state y definicion de acciones
    actions=[]
    state = 0 
    location_to_state = dict()
    for y, ele in enumerate(imgpad_edos):
        for x, sub in enumerate(ele): 
            if sub != 0:
                location_to_state.setdefault(str(int(sub)),int(state))
                actions.append(int(state))
                state += 1
    
    ################ Crear diccionario de coordenadas
    
    #enumerate con pasos negativos, para el diccionario
    def enumerate2(xs, start=0, step=1):
        for x in xs:
            yield (start, x)
            start += step
    
    #diccionario 'localizacion' = {coordenadas} 20x20
    if resolution == 20:
        location_to_coordinate = dict()
        for j, ele in enumerate2(edos, start=10,step=-1):
            for idx, sub in enumerate(ele, start=-10):
                location_to_coordinate.setdefault(str(sub),(idx, j))
                
    #diccionario 'localizacion' = {coordenadas} 40x40
    if resolution == 40:
        location_to_coordinate = dict()
        for j, ele in enumerate2(edos, start=10,step=-0.5):
            for idx, sub in enumerate2(ele, start=-10,step=0.5):
                location_to_coordinate.setdefault(str(sub),(idx, j))
                
    #diccionario 'localizacion' = {coordenadas} 100x100
    if resolution == 100:
        location_to_coordinate = dict()
        for j, ele in enumerate2(edos, start=25,step=-0.5):
            for idx, sub in enumerate2(ele, start=-25,step=0.5):
                location_to_coordinate.setdefault(str(sub),(idx, j))
                            
            
    ############################ Costruir Matriz de adyacencia
    
    imgpad_mask = np.array(imgpad_edos, dtype=bool)
    adjacency = skf.image.img_to_graph(img=imgpad_edos, mask=imgpad_mask, return_as=np.ndarray)
    
    #Cambiar la diagonal principal por ceros y normalizar todos los valores a 1
    R = np.zeros(adjacency.shape) #matriz de recompensa
    
    for y, ele in enumerate(adjacency):
        for x, sub in enumerate(ele): 
            if x != y:
                if sub > 1:
                    sub = 1
                R[y, x] = sub
    
    ############################ Agregar diagonales a Matriz de adyacencia

    #Ciclo para aplicar agregar diagonales comparando con matriz_edos
    if diagonals == 1:
        for y, ele in enumerate(imgpad_edos):
            for x, sub in enumerate(ele):
                if imgpad_edos[y, x] > 0:
                    edo = imgpad_edos[y, x]
                    location_axis = int(location_to_state[str(int(edo))])
                    #de momento solo las esquinas porque ya tengo lo demas con la funcion de sklearn
                    if x < resolution and y > 0:
                        if imgpad_edos[y-1, x+1] > 0:
                            edo = imgpad_edos[y-1, x+1] #esquina superior-derecha
                            location = int(location_to_state[str(int(edo))])
                            R[location_axis, location] = 1
                    if x > 0 and y > 0:
                        if imgpad_edos[y-1, x-1] > 0:
                            edo = imgpad_edos[y-1, x-1] #esquina superior-izquierda
                            location = int(location_to_state[str(int(edo))])
                            R[location_axis, location] = 1
                    if x < resolution and y < resolution:
                        if imgpad_edos[y+1, x+1] > 0:
                            edo = imgpad_edos[y+1, x+1] #esquina inferior-derecha
                            location = int(location_to_state[str(int(edo))])
                            R[location_axis, location] = 1
                    if x > 0 and y < resolution:
                        if imgpad_edos[y+1, x-1] > 0:
                            edo = imgpad_edos[y+1, x-1] #esquina inferior-izquierda
                            location = int(location_to_state[str(int(edo))])
                            R[location_axis, location] = 1
    
    ############################ Customisar recompensa con logica de muros
    
    custom_R = imgpad_edos.astype(float).copy()#Matriz que guarda la recompensa customisada
    ventana = densidad*2+1
    densidad+=1
    if custom_reward == 1:
        for y, ele in enumerate(imgpad_edos):
            for x, sub in enumerate(ele):
                if imgpad_edos[y, x] > 0:
                    count_zero = 0
                    for n in range(1,densidad):
                        if diagonals == 1:
                            if x <= resolution-n and y >= n:
                                if imgpad_edos[y-n, x+n] == 0: #esquina superior-derecha
                                    count_zero += 1
                            if x >= n and y >= n:
                                if imgpad_edos[y-n, x-n] == 0:  #esquina superior-izquierda
                                    count_zero += 1
                            if x <= resolution-n and y <= resolution-n:
                                if imgpad_edos[y+n, x+n] == 0: #esquina inferior-derecha
                                    count_zero += 1
                            if x >= n and y <= resolution-n:
                                if imgpad_edos[y+n, x-n] == 0: #esquina inferior-izquierda
                                    count_zero += 1
                        if y >= n:
                            if imgpad_edos[y-n, x] == 0: #arriba
                                count_zero += 1
                        if y <= resolution-n:
                            if imgpad_edos[y+n, x] == 0: #abajo
                                count_zero += 1
                        if x <= resolution-n:
                            if imgpad_edos[y, x+n] == 0: #enfrente
                                count_zero += 1
                        if x >= n:
                            if imgpad_edos[y, x-n] == 0: #atras
                                count_zero += 1          
                    posibles_zeros = ventana**2-densidad #area de la ventana menos la densidad(representa un camino al centro)
                    if ventana == 1:
                        recompensa = 2
                    else:
                        recompensa = 2 - (count_zero/posibles_zeros)
                    custom_R[y, x]=recompensa

    #Ciclo para agregar recompensa con logica de muros
        for y, ele in enumerate(imgpad_edos):
            for x, sub in enumerate(ele):
                if imgpad_edos[y, x] > 0:
                    edo = imgpad_edos[y, x]
                    location_axis = int(location_to_state[str(int(edo))])
                    if diagonals == 1:
                        if x < resolution and y > 0:
                            if imgpad_edos[y-1, x+1] > 0:
                                edo = imgpad_edos[y-1, x+1] #esquina superior-derecha
                                location = int(location_to_state[str(int(edo))])
                                R[location_axis, location] = custom_R[y-1, x+1]
                        if x > 0 and y > 0:
                            if imgpad_edos[y-1, x-1] > 0:
                                edo = imgpad_edos[y-1, x-1] #esquina superior-izquierda
                                location = int(location_to_state[str(int(edo))])
                                R[location_axis, location] = custom_R[y-1, x-1]
                        if x < resolution and y < resolution:
                            if imgpad_edos[y+1, x+1] > 0:
                                edo = imgpad_edos[y+1, x+1] #esquina inferior-derecha
                                location = int(location_to_state[str(int(edo))])
                                R[location_axis, location] = custom_R[y+1, x+1]
                        if x > 0 and y < resolution:
                            if imgpad_edos[y+1, x-1] > 0:
                                edo = imgpad_edos[y+1, x-1] #esquina inferior-izquierda
                                location = int(location_to_state[str(int(edo))])
                                R[location_axis, location] = custom_R[y+1, x-1]  
                    if y > 0:
                        if imgpad_edos[y-1, x] > 0:
                            edo = imgpad_edos[y-1, x] #arriba
                            location = int(location_to_state[str(int(edo))])
                            R[location_axis, location] = custom_R[y-1, x]
                    if y < resolution:
                        if imgpad_edos[y+1, x] > 0:
                            edo = imgpad_edos[y+1, x] #abajo
                            location = int(location_to_state[str(int(edo))])
                            R[location_axis, location] = custom_R[y+1, x]
                    if x < resolution:
                        if imgpad_edos[y, x+1] > 0:
                            edo = imgpad_edos[y, x+1] #enfrente
                            location = int(location_to_state[str(int(edo))])
                            R[location_axis, location] = custom_R[y, x+1]
                    if x > 0:
                        if imgpad_edos[y, x-1] > 0:
                            edo = imgpad_edos[y, x-1] #atras
                            location = int(location_to_state[str(int(edo))])
                            R[location_axis, location] = custom_R[y, x-1]        
    
    return actions, location_to_state, location_to_coordinate, R, imgpad_edos, custom_R

#linea debug
# image = 'Oficina.png'#'Oficina.png'#"SuperMercado.png"#'Farmacia100x100pxl.png'#"Mapa40x40pxl.png"
# resolution=100#100
# diagonals= 0 #1=si, 0=no 
# custom_reward= 0 #1=si, 0=no
# densidad = 0 # ventana=2*densidad+1
# actions, location_to_state, location_to_coordinate, R, imgpad_edos, custom_R = building_the_environment(image, resolution, diagonals, custom_reward, densidad)
