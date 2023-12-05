#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:18:09 2023

@author: Raydesel
"""

import numpy as np
from building_the_environment import *
import os
import time
import random
import skfuzzy as fuzz
import csv
import pandas as pd

#actions, location_to_state, location_to_coordinate, R = builing_the_environment('Mapa20x20pxl.png')

def training_thompson(iterations, Q, R_new, actions, densidad, gamma, alpha):
    
    for n in range(iterations):
        current_state = np.random.randint(0, len(actions))
        playable_actions = np.where(R_new[current_state, :] > 0)[0]
        max_random = 0
        ventana = densidad*2+1
        posibles_zeros = ventana**2 - densidad
        
        for action_index in playable_actions:
            reward = R_new[current_state, action_index]
            
            if reward > 2:
                n_zeros = 0
                n_ones = posibles_zeros
            else:
                n_zeros = (2 - reward) / (1 / posibles_zeros)
                n_ones = posibles_zeros - n_zeros
                
            random_beta = random.betavariate(n_ones + 1, n_zeros + 1)
            
            if random_beta > max_random:
                max_random = random_beta
                next_state = action_index
        
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    
    return Q


def training_gauss(iterations, Q, R_new, actions, gamma, alpha, sigma):
    
    for n in range(iterations):
        current_state = np.random.randint(0,len(actions))
        # Creación de la distribución fuzzy
        mean = np.argmax(R_new[current_state,])
        fuzzy_vals = fuzz.gaussmf(np.arange(0, len(R_new[current_state,])), mean, sigma)
        # Aplicación de la distribución a la matriz Q
        fuzzy_Q = R_new[current_state,] * fuzzy_vals
        # Normalización de la distribución fuzzy para que sume 1
        fuzzy_Q = fuzzy_Q / np.sum(fuzzy_Q)
        # Obtiene un estado aleatorio tomando en cuenta la probabilidad fuzzy
        next_state = np.random.choice(np.arange(0, len(R_new[current_state,])), p=fuzzy_Q)
        #next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    return Q

def training(iterations, Q, R_new, actions, gamma, alpha):
    
    for i in range(iterations):
        current_state = np.random.randint(0, len(actions))
        playable_actions = np.where(R_new[current_state, :] > 0)[0]
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
        
    return Q

def inference(starting_location, ending_location, location_to_state, state_to_location, Q):
    route = [starting_location]
    next_location = starting_location
    start_time = time.time()  # Tiempo de inicio del bucle
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
        
        elapsed_time = time.time() - start_time  # Tiempo transcurrido desde el inicio del bucle
        #print(elapsed_time)
        if elapsed_time >= 10:  # Si han pasado 10 segundos
            route = []
            break
    return route


# Making a function that returns the shortest route from a starting to ending location
def route(actions, location_to_coordinate, location_to_state, R, starting_location, ending_location, resolution, densidad=0, gamma=0.75, alpha=0.9, iterations=200000, Q_table=0, entrenamiento=0):
    
    #Making a mapping from the states to the locations
    state_to_location = {state: location for location, state in location_to_state.items()}

    Q_convergencia = 0
    R_new = R.copy()
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    #Cargar o Crear Q_table
    if os.path.exists('Q_table.npy') and Q_table==1:
        # Cargar la matriz desde el archivo
        Q = np.load('Q_table.npy')
        #print("Q_table Cargada")
    else:
        #inicializar una nueva Q_table
        Q = np.zeros_like(R)  # Inicializar una nueva Q-table con cero
        #print("Q_table Creada")
    
    contador = 0 #numero de entrenamientos 
    route = []
    start_time = time.time()
    # Crear un DataFrame para almacenar los datos
    data = []
    
    if resolution == 40:
        sigma = 50
    elif resolution == 100:
        sigma = 100

    while not route:
        contador+=1
        #print(f'iniciando entrenamiento {contador}')
        if entrenamiento == 0:
            old_Q = np.copy(Q)
            Q = training(iterations, Q, R_new, actions, gamma, alpha)
            # Calcula la diferencia promedio entre los valores antiguos y nuevos de la Q-table
            diff = np.abs(Q - old_Q).mean()
            #print(diff)
            # Guardar los valores en el DataFrame
            data.append([contador, diff])
            # Si la diferencia es menor que un umbral determinado, considera que la Q-table ha convergido
            if diff == 0.0:
                #print("La Q-table ha convergido")
                Q_convergencia = 1
                
        elif entrenamiento == 1:
            old_Q = np.copy(Q)
            Q = training_thompson(iterations, Q, R_new, actions, densidad, gamma, alpha)
            # Calcula la diferencia promedio entre los valores antiguos y nuevos de la Q-table
            diff = np.abs(Q - old_Q).mean()
            #print(diff)
            # Guardar los valores en el DataFrame
            data.append([contador, diff])
            # Si la diferencia es menor que un umbral determinado, considera que la Q-table ha convergido
            if diff == 0.0:
                #print("La Q-table ha convergido")
                Q_convergencia = 1
                
        elif entrenamiento == 2:
            # Actualiza los valores antiguos de la Q-table para el próximo cálculo de diferencia
            old_Q = np.copy(Q)
            Q = training_gauss(iterations, Q, R_new, actions, gamma, alpha, sigma)
            # Calcula la diferencia promedio entre los valores antiguos y nuevos de la Q-table
            diff = np.abs(Q - old_Q).mean()
            #print(diff)
            # Guardar los valores en el DataFrame
            data.append([contador, diff])
            # Si la diferencia es menor que un umbral determinado, considera que la Q-table ha convergido
            if diff == 0.0:
                #print("La Q-table ha convergido")
                Q_convergencia = 1
        
        #np.save('Q_table.npy', Q)
        
        if Q_convergencia == 1:
            route = inference(starting_location, ending_location, location_to_state, state_to_location, Q)

    #traduce las localizaciones de la ruta obtendida a coordenadas 
    route_to_coordinate=[]
    for location in route:
        route_to_coordinate.append(location_to_coordinate[location])
        
    # Crear el DataFrame a partir de los datos
    df_data_diff = pd.DataFrame(data, columns=['Epoca', 'Diferencia'])
        
    end_time = time.time()
    total_time = (end_time - start_time)/60.0
    print("Minutos de entrenamiento:", total_time)
    return df_data_diff, diff, total_time, contador, route_to_coordinate, Q
# Busca la ruta optima entre una localizacion inicial y una final
#route_to_coordinate = route(actions, location_to_state, R, starting_location = '45', ending_location = '313', gamma=0.75, alpha=0.9, iterations=200000)
