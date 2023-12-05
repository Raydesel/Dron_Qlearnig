#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:39:11 2023

@author: Raydesel

Generador de trayectorias utilizando Aprendizaje Q
 y metodos de interpolacion
 
Se ejecuta utilizando multiprocessing para entrenar varias tablas Q al mismo tiempo
"""
from building_the_environment import *
from Aprendizaje_Q import *
from bspline_with_order import *
from bezier import *
from delay_z_yaw import *
from RMSE_bs1_bs10 import *
import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import traceback
import itertools
import multiprocessing

def add_giro(last_yaw, b):
    first_row = np.copy(b[0,:])
    first_row[0]=2.0#delay para girar
    start_yaw = b[0,4]
    rows = []
    yaw_flag=last_yaw
    while (not(-45 <= (last_yaw-start_yaw) <= 45)):
        if yaw_flag > 0 and (last_yaw-start_yaw) > 45:
            last_yaw=last_yaw-45
            first_row[-1]=last_yaw
            rows.append(np.copy(first_row))
        if yaw_flag < 0 and (last_yaw-start_yaw) < -45:
            last_yaw=last_yaw+45
            first_row[-1]=last_yaw
            rows.append(np.copy(first_row))          
    b = np.insert(b, 0, rows, axis=0)
    return b

def extend_route(actions, location_to_coordinate, location_to_state, R, starting_location, intermediary_locations, ending_location, gamma, alpha, iterations, Q_table, delay):
    count_coord = []
    route_ = route(actions, location_to_coordinate, location_to_state, R, starting_location, intermediary_locations[0], gamma, alpha, iterations, Q_table)
    count_coord.append(len(route_))
    e = route_ 
    data_bspline = bspline_with_order(3, 100, route_)
    delay = (len(route_)*0.08)/26
    a = savedata(data_bspline, delay)
    last_yaw = a[-1,4]
    for i in range(1, len(intermediary_locations)):
        route_ = route(actions, location_to_coordinate, location_to_state, R, intermediary_locations[i-1], intermediary_locations[i], gamma, alpha, iterations, Q_table)
        count_coord.append(len(route_))
        e += route_
        data_bspline = bspline_with_order(3, 100, route_)
        delay = (len(route_)*0.08)/26
        b = savedata(data_bspline, delay)
        b = add_giro(last_yaw, b)
        last_yaw = b[-1,4]
        a = np.concatenate((a, b), axis=0)
    route_ = route(actions, location_to_coordinate, location_to_state, R, intermediary_locations[-1], ending_location, gamma, alpha, iterations, Q_table)
    count_coord.append(len(route_))
    e += route_
    data_bspline = bspline_with_order(3, 100, route_)
    delay = (len(route_to_coordinate)*0.08)/26
    b = savedata(data_bspline, delay)
    b = add_giro(last_yaw, b)
    a = np.concatenate((a, b), axis=0)
    return a, e, count_coord

def guardar(df_data_diff,bs1,RMSE_xy,percent_error_xy,ventana,resolution, metodo, path_metodo, name, delay_x_y_z_yaw, route_to_coordinate, starting_location, ending_location, Q, imagen):

    #Especificar las coordenadas x e y del spline
    x_y = pd.DataFrame(delay_x_y_z_yaw)
    x_y = np.asarray(x_y)
    x = x_y[:,1] 
    y = x_y[:,2]
    data = [x,y]
    #Especificar las coordenadas x e y de generadas con RL
    x_y = pd.DataFrame(route_to_coordinate)
    x_y = np.asarray(x_y)
    x = x_y[:,0] 
    y = x_y[:,1]
    xy = [x,y]
    
    #Gráfica
    plt.figure(1, figsize=(8,8)) #aumentar tamaño de la figura
    """
    This function is responsible for displaying obstacles on plot.
    """
    imagen = '(gz)'+imagen #'(gz)Farmacia100x100pxl.png'
    image = cv2.imread(imagen, 0)
    mapa = np.where(image >= 0.5, 1, 0)

    # Convertir la matriz binaria en una lista de tuplas con las coordenadas de los obstáculos
    obstaculos = []
    for i in range(len(mapa)):
        for j in range(len(mapa[i])):
            if mapa[i][j] == 0:
                if resolution == 40:
                    x = -10 + j*0.5
                    y = 10 - i*0.5
                elif resolution == 100:
                    x = -25 + j*0.5
                    y = 25 - i*0.5
                else:
                    print('linea 111: resolucion no indicada')
                obstaculos.append((x, y))
                
    # Dibujar los obstáculos
    for obs in obstaculos:
        obs_x = [obs[0], obs[0]+0.5, obs[0]+0.5, obs[0], obs[0]]
        obs_y = [obs[1], obs[1], obs[1]-0.5, obs[1]-0.5, obs[1]]
        #plt.fill(obs_x, obs_y, "gray")
        plt.fill(obs_x, obs_y, "black", alpha=0.2)
        
    #loop para aumentar el indice 
    txt_id=0
    while os.path.exists(path_metodo+name+str(txt_id)):
        txt_id += 1
    # Ruta para crear la carpeta
    path = path_metodo+name+str(txt_id)
    # Llamada a la función para crear la carpeta
    crear_carpeta_si_no_existe(path)
    # Ruta para crear la carpeta
    path = path_metodo+name+str(txt_id)+'/'
    
    plt.plot(xy[0],xy[1],'*r')
    plt.plot(data[0],data[1],'k')
    plt.title(f"%error: {percent_error_xy:.3f} RMSE: {RMSE_xy:.3f} Numero de estados: {len(route_to_coordinate)}, ventana: {ventana}")
    path_img = path_metodo+name+str(txt_id)
    plt.savefig(path_img+'.jpg')
    plt.show()
    
    np.savetxt(path+name+str(txt_id)+'.txt', delay_x_y_z_yaw, delimiter='    ', fmt='%.4f')
    np.savetxt(path+name+str(txt_id)+'.csv', delay_x_y_z_yaw, delimiter=', ', fmt='%.4f')
    
    np.savetxt(path+name+str(txt_id)+'_bs1.csv', delay_x_y_z_yaw, delimiter=', ', fmt='%.4f')
    
    #Guardar la Q_table
    np.save(path+name+str(txt_id)+'_Q_table_.npy', Q)
    
    #Guardar la diferencia entre Qtables para cada epoca
    nombre_archivo = path+'Q_diff_'+str(txt_id)+'.csv'
    df_data_diff.to_csv(nombre_archivo, mode='w', index=False)

    print(name+"coordenadas"+str(txt_id))
    return path, txt_id 


def crear_carpeta_si_no_existe(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        print("La carpeta se ha creado correctamente.")
    else:
        print("La carpeta ya existe.")
        
def varios_delays(data, delay, extend, path, txt_id, name):
    for retardo in [50, 40, 30, 20, 15, 12, 10, 9, 8, 7]:
        retardo_txt = retardo/1000
        delay_x_y_z_yaw = savedata(data, retardo_txt, extend)
        np.savetxt(path+name+str(txt_id)+'_'+str(retardo)+'ms.txt', delay_x_y_z_yaw, delimiter='    ', fmt='%.4f')

def Calcula_RMSE_bs1_bs10(bs10, points, route_to_coordinate, delay, extend):
    data_bspline = bspline_with_order(1, points, route_to_coordinate[:-1]) #bs10 queda atras de la ultima coordenada
    delay_x_y_z_yaw = savedata(data_bspline, delay, extend)
    bs1 = delay_x_y_z_yaw
    RMSE_xy, percent_error_xy = calcula_error(bs1, bs10)
    return RMSE_xy, percent_error_xy, data_bspline

def crear_save_df(name, txt_id, path_folder, imagen, time_train, total_time, starting_location, ending_location, metodo, route_to_coordinate, gamma, alpha, count_iterations, delay, densidad, diagonals, convergencia, points, bspline, order, bezier, RMSE_xy, percent_error_xy):
    
    inicio_final= str(starting_location)+'-'+str(ending_location)
    count_edos = len(route_to_coordinate)
    metros = count_edos/2
    ventana = densidad*2+1 
    ventana = str(ventana)+'x'+str(ventana)
    
    # Crear DataFrame
    data_trayectoria = {
        'Path': path_folder,
        'Entorno': imagen[:-4],
        
        'Metodo': metodo,
        'Ventana': ventana,
        'Diagonales': diagonals,
        'Gamma(descuento)': gamma,
        'Alpha(aprendizaje)': alpha,
        'Retardos(ms)': '50, 40, 30, 20, 15, 12, 10, 9, 8, 7',
        
        'Tiempo_entrenamiento(min)': time_train,
        'Tiempo_total(min)': total_time,
        'Iteraciones': count_iterations,
        'Convergencia': convergencia,
        
        'Inicio-Final(edos)': inicio_final,
        '#edos': count_edos,
        'Distancia(m)': metros,
        
        'Puntos_interpolacion': points,
        'Bspline': bspline,
        'Grado_bspline': order,
        'Bezier': bezier,
        
        'RMSE_xy': RMSE_xy,
        'Percent_error_xy': percent_error_xy
        }
    
    df_trayectoria = pd.DataFrame(data_trayectoria, index=range(1))
        
    # Guardar el DataFrame en un archivo CSV
    nombre_archivo = path_folder+name+str(txt_id)+'_datos.csv'
    df_trayectoria.to_csv(nombre_archivo, mode='w', index=False)
    return df_trayectoria

def final_codigo(path_inicio, df_concatenado):
    #loop para aumentar el indice
    name='Trayectorias_datos_'
    id_df=0
    while os.path.exists(path_inicio+name+str(id_df)+'.csv'):
        id_df += 1
    # Guardar el DataFrame en un archivo CSV
    nombre_archivo = path_inicio+'Trayectorias_datos_'+str(id_df)+'.csv'
    df_concatenado.to_csv(nombre_archivo, mode='w', index=False)

def process_combination(combination):
    
    #PARAMETROS
    imagen = 'Farmacia.png'#'Bodega.png' #'Oficina.png'#'SuperMercado.png'#'Casa.png'
    resolution = 100 #40
    iterations= (resolution+1)*(resolution+1)#100000#300000 #interaciones por entrenamiento de la Qtabla
    gamma=0.99 #0.75 factor de descuento 0.9
    alpha=0.9 #0.9 factor de aprendizaje
    delay=0.015
    #Q_table = 0 #1=cargar, 0=crear
    extend = 0 #1=si, 0=no
    # intermediary_location = ['251', '1481', '1513']
    #Orden del betaspline
    interpolation_bspline = 1 #1=si, 0=no
    order = 10#10
    points = 1000 #100 #200 #300 Puntos spline
    interpolation_bezier = 0 #1=si, 0=no
    
    path = '/home/ariel/catkin_ws/src/rotors_simulator/rotors_gazebo/resource/waypoints/'
    # Ruta para crear la carpeta
    path_inicio = path+imagen[:-4]+'/'
    # Llamada a la función para crear la carpeta
    crear_carpeta_si_no_existe(path_inicio)
    
    count_iterations = 0
    contador_acumulado = 0
    route_convergence = []
    Q_table = 0 #0:inicia creando una nueva Q_table 1:Carga una Q_table
    df_concatenado = pd.DataFrame() #Guarda el df de las trayectorias
    
    ruta, diagonals, densidad, entrenamiento = combination
    starting_location = ruta[0]
    ending_location = ruta[1]

    print(f'edo_inicio:{starting_location} a edo_final:{ending_location}')
    edoinicio_edofinal = str(starting_location) + '-' + str(ending_location)

    if densidad > 1:
        custom_reward = 1
    else:
        custom_reward = 0
    ventana = densidad * 2 + 1

    # BUILDING THE ENVIRONMENT
    actions, location_to_state, location_to_coordinate, R, img_edos, custom_R = building_the_environment(imagen, resolution, diagonals, custom_reward, densidad)

    if entrenamiento == 0:
        metodo = 'Metodo_random'
        print(metodo)
        # Ruta para crear la carpeta
        path = path_inicio+metodo+'/'+edoinicio_edofinal+'/'
        # Llamada a la función para crear la carpeta
        crear_carpeta_si_no_existe(path)
    elif entrenamiento == 1:
        metodo = 'Metodo_thompson_sampling'
        print(metodo)
        # Ruta para crear la carpeta
        path = path_inicio+metodo+'/'+edoinicio_edofinal+'/'
        # Llamada a la función para crear la carpeta
        crear_carpeta_si_no_existe(path)
    elif entrenamiento == 2:
        metodo = 'Metodo_gauss'
        print(metodo)
        # Ruta para crear la carpeta
        path = path_inicio+metodo+'/'+edoinicio_edofinal+'/'
        # Llamada a la función para crear la carpeta
        crear_carpeta_si_no_existe(path)


    # BUILDING THE AI SOLUTION WITH Q-LEARNING
    # Busca la ruta optima entre una localizacion inicial y una final
    if extend == 0:
        start_time = time.time()
        df_data_diff, convergencia, time_train, contador, route_to_coordinate, Q = route(actions, location_to_coordinate, location_to_state, R, starting_location, ending_location, resolution, densidad, gamma, alpha, iterations, Q_table, entrenamiento)

        #interpolacion usando B-spline_with_order
        if interpolation_bspline == 1:
            name = 'coordenadas_bspline_'
            data_bspline = bspline_with_order(order, points, route_to_coordinate)
            delay_x_y_z_yaw = savedata(data_bspline, delay, extend)
            RMSE_xy, percent_error_xy, bs1 = Calcula_RMSE_bs1_bs10(delay_x_y_z_yaw, points, route_to_coordinate, delay, extend)
            path_folder, txt_id = guardar(df_data_diff, bs1, RMSE_xy, percent_error_xy,ventana,resolution, metodo, path, name, delay_x_y_z_yaw, route_to_coordinate, starting_location, ending_location, Q, imagen)
            varios_delays(data_bspline, delay, extend, path_folder, txt_id, name)
            
            
        if interpolation_bezier == 1:
            name = 'coordenadas_bezier_'
            data_bezier = bezier(route_to_coordinate, points)
            delay_x_y_z_yaw = savedata(data_bezier, delay, extend)
            path_folder, txt_id = guardar(percent_error_xy,ventana,resolution, metodo, path, name, delay_x_y_z_yaw, route_to_coordinate, starting_location, ending_location, Q, imagen)
            varios_delays(data_bezier, delay, extend, path_folder, txt_id, name)
            
        contador_acumulado += contador
        count_iterations = (contador_acumulado)*iterations
        end_time = time.time()
        total_time = (end_time - start_time)/60.0
        print("Tiempo total de ejecución: ", total_time)
        print(f'Numero de entrenaminetos = {contador_acumulado}\nNumero de iteraciones = {count_iterations}')
        
        df_trayectoria = crear_save_df(name, txt_id, path_folder, imagen, time_train, total_time, starting_location, ending_location, metodo, route_to_coordinate, gamma, alpha, count_iterations, delay, densidad, diagonals, convergencia, points, interpolation_bspline, order, interpolation_bezier, RMSE_xy, percent_error_xy)
        df_concatenado = pd.concat([df_concatenado, df_trayectoria], ignore_index=True)

    else:
        delay_x_y_z_yaw, route_to_coordinate, count_coord = extend_route(actions, location_to_coordinate, location_to_state, R, starting_location, intermediary_location, ending_location, gamma, alpha, iterations, Q_table, delay)
        name = 'coordenadas_bspline_'
        guardar(path, name, delay_x_y_z_yaw, route_to_coordinate, starting_location, ending_location, Q, imagen)
        
    if entrenamiento == 0:
        np.save('Q_table_random.npy', Q)
    elif entrenamiento == 1:
        np.save('Q_table_thompson.npy', Q)
    elif entrenamiento == 2:
        np.save('Q_table_gauss.npy', Q)
    
    final_codigo(path, df_concatenado)    

if __name__ == '__main__':
    combinations = []
    farmacia_rutas = [('9029', '4567'), ('4084', '9609'), ('6270', '5637'), ('6914', '6951')]
    #mercado_rutas = [('4739', '3135'), ('3587', '3943'), ('6394', '3412'), ('5293', '3162')]
    #bodega_rutas = [('1549','7760'),('1869','7604'),('1309','5786'),('1722','5356')]
    #oficina_rutas = [('6577','6751'),('3256','6923'),('3502','7015'),('3424','3247')]
    #casa_rutas = [('1', '1528'), ('866', '1534'), ('445', '1522'), ('1475', '1102')]
    
    # for ruta in farmacia_rutas:
    #     starting_location, ending_location = ruta
    #     combination = list(itertools.product([(starting_location, ending_location)], [0, 1], [0, 1, 2, 3], [0, 1, 2]))
    #     combinations.extend(combination)
    
    combinations = [(('9029', '4567'), 0, 0, 1),
                    (('9029', '4567'), 0, 0, 2),
                    (('9029', '4567'), 0, 2, 1),
                    (('9029', '4567'), 0, 2, 2),
                    (('9029', '4567'), 0, 3, 1),
                    (('9029', '4567'), 0, 3, 2),
                    (('9029', '4567'), 1, 0, 1),
                    (('9029', '4567'), 1, 0, 2),
                    (('9029', '4567'), 1, 2, 1),
                    (('9029', '4567'), 1, 2, 2),
                    (('9029', '4567'), 1, 3, 1),
                    (('9029', '4567'), 1, 3, 2),
                    (('9029', '4567'), 0, 0, 0),
                    (('9029', '4567'), 0, 1, 0),
                    (('9029', '4567'), 0, 2, 0),
                    (('9029', '4567'), 0, 3, 0),
                    (('9029', '4567'), 1, 0, 0),
                    (('9029', '4567'), 1, 1, 0),
                    (('9029', '4567'), 1, 2, 0),
                    (('9029', '4567'), 1, 3, 0)]
    
    with multiprocessing.Pool() as pool:
        pool.map(process_combination, combinations)
    
    # # Crear un grupo de procesos utilizando multiprocessing.Pool() con un bloque with
    # with multiprocessing.Pool() as pool:
    #     # Ejecutar cada combinación en un proceso separado utilizando apply_async()
    #     results = [pool.apply_async(process_combination, (c,)) for c in combinations]
    
        # # Recopilar y mostrar los resultados individualmente
        # for result in results:
        #     result.get()
            
        # for i, result in enumerate(results):
        #     result.get()
        #     print(f"Proceso {i+1}/{len(combinations)} completado")
            


