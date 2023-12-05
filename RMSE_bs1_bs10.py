#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:43:11 2023

@author: Raydesel
"""
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import matplotlib.pyplot as plt
import cv2


def cargar_archivos_txt(txt):
    fileID = np.loadtxt(txt)
    coord_x_spline = fileID[0:,1]
    coord_y_spline = fileID[0:,2]
    yaw_spline = fileID[0:,4]
    return coord_x_spline, coord_y_spline, yaw_spline

def cargar_archivos_xy_yaw(delay_x_y_z_yaw):
    coord_x_spline = delay_x_y_z_yaw[0:,1]
    coord_y_spline = delay_x_y_z_yaw[0:,2]
    yaw_spline = delay_x_y_z_yaw[0:,4]
    return coord_x_spline, coord_y_spline, yaw_spline

#Porcentaje de error conciderando valores true
def calculate_percent_error(sum_abs_err, sum_coord):
    if sum_coord == 0:
        return float('inf')
    else:
        return (sum_abs_err / sum_coord) * 100

def calcula_error(bs1, bs10):
    #Calcular error entre Bspline de grado 1(true) y grado 10(predicted)
    #Genera %error, RMSE y MSE para x, y, xy, yaw, de momento solo retorno rmse_xy porque es el unico requerido
    # coord_x_spline1, coord_y_spline1, yaw_spline1 = cargar_archivos_txt(bs1)
    
    # coord_x_spline10, coord_y_spline10, yaw_spline10 = cargar_archivos_txt(bs10)
    
    coord_x_spline1, coord_y_spline1, yaw_spline1 = cargar_archivos_xy_yaw(bs1)
    
    coord_x_spline10, coord_y_spline10, yaw_spline10 = cargar_archivos_xy_yaw(bs10)

    
    if len(coord_x_spline1) > len(coord_x_spline10): 
        diff_length = len(coord_x_spline10)-len(coord_x_spline1) #bs10 genera menos puntos al final
        coord_x_spline1 = coord_x_spline1[:diff_length]  
        coord_y_spline1 = coord_y_spline1[:diff_length] 
        yaw_spline1 = yaw_spline1[:diff_length]
    elif len(coord_x_spline10) > len(coord_x_spline1):
        diff_length = len(coord_x_spline1)-len(coord_x_spline10) #bs1 genera menos puntos al final
        coord_x_spline10 = coord_x_spline10[:diff_length]  
        coord_y_spline10 = coord_y_spline10[:diff_length] 
        yaw_spline10 = yaw_spline10[:diff_length]
        
    # CALCULA RMSE

    #Calculo de mse(true, predicted)
    err_x = mse(coord_x_spline1,coord_x_spline10) #error eje x
    err_y = mse(coord_y_spline1,coord_y_spline10) #error eje y
    err_yaw = mse(yaw_spline1,yaw_spline10) #error yaw
    
    rmse_x = np.sqrt(err_x)
    rmse_y = np.sqrt(err_y)
    rmse_yaw = np.sqrt(err_yaw)
    
    #MSE considerando x e y
    r_x = (coord_x_spline10 - coord_x_spline1)**2
    L = len(r_x) #tamaño de r_x
    r_y = (coord_y_spline10 - coord_y_spline1)**2
    MSEr = sum(r_x + r_y)/L #calculo del error
    
    RMSEr = np.sqrt(MSEr)
    
    # CALCULA %ERROR
    sum_abs_err_xy = np.abs(sum(coord_x_spline10 - coord_x_spline1)) + np.abs(sum(coord_y_spline10 - coord_y_spline1))
    sum_coord_xy = np.abs(sum(coord_x_spline1)) + np.abs(sum(coord_y_spline1))
    percent_error_xy = calculate_percent_error(sum_abs_err_xy, sum_coord_xy)
    
    sum_abs_err_yaw = np.abs(sum(yaw_spline10 - yaw_spline1))
    sum_coord_yaw = np.abs(sum(yaw_spline1))
    percent_error_yaw = calculate_percent_error(sum_abs_err_yaw, sum_coord_yaw)
    
    #tamano del arreglo 'predicted'
    N = len(coord_x_spline10)
    
    #tamano del arreglo 'true'
    n_coord = len(coord_x_spline1);
    
    #genera vector de puntos para evaluar spline_true
    tx = np.linspace(0,N-1,n_coord)
    
    
    fig = plt.figure(8, figsize=(8,8))
    plt.plot(tx,coord_x_spline10, label='X observado'); #predicted x
    plt.xlabel('Tiempo')
    plt.ylabel('Eje X')
    plt.plot(tx,coord_x_spline1,'r', label='X esperado') #true x
    plt.legend()
    
    fig = plt.figure(9, figsize=(8,8))
    plt.plot(tx,coord_y_spline10, label='Y observado'); #predicted y
    plt.xlabel('Tiempo')
    plt.ylabel('Eje Y')
    plt.plot(tx,coord_y_spline1,'r', label='Y esperado') #true y
    plt.legend()
    
    # fig = plt.figure(10, figsize=(8,8))
    # """
    # # This function is responsible for displaying obstacles on plot.
    # # """
    # # Cargar binarymap.png a matriz numpy
    # image = 'Mapa40x40pxl.png'
    # image = cv2.imread(image, 0)
    # mapa = np.where(image >= 0.5, 1, 0)
    
    # # Convertir la matriz binaria en una lista de tuplas con las coordenadas de los obstáculos
    # obstaculos = []
    # for ii in range(len(mapa)):
    #     for j in range(len(mapa[ii])):
    #         if mapa[ii][j] == 0:
    #             x = -10 + j*0.5
    #             y = 10 - ii*0.5
    #             obstaculos.append((x, y))
    
    # # Dibujar los obstáculos
    # for obs in obstaculos:
    #     obs_x = [obs[0], obs[0]+0.5, obs[0]+0.5, obs[0], obs[0]]
    #     obs_y = [obs[1], obs[1], obs[1]-0.5, obs[1]-0.5, obs[1]]
    #     #plt.fill(obs_x, obs_y, "gray")
    #     plt.fill(obs_x, obs_y, "black", alpha=0.2)
    
    # #predicted xy con línea más gruesa
    # plt.plot(coord_x_spline10,coord_y_spline10, linewidth=1.5, label='Trayectoria observada');
    # # Seleccione cada 5 punto de las coordenadas x e y de la trayectoria esperada
    # coord_x_spline_sub = coord_x_spline1[::5]
    # coord_y_spline_sub = coord_y_spline1[::5]
    # # Grafique los puntos seleccionados con tamaño de punto 10 y transparencia 0.5
    # plt.scatter(coord_x_spline_sub, coord_y_spline_sub, s=10, alpha=0.5, color='r', label='Trayectoria esperada')
    # plt.xlabel('Eje X')
    # plt.ylabel('Eje Y')
    # plt.legend()
    
    # fig = plt.figure(11, figsize=(8,8))
    # plt.plot(tx,yaw_spline10, label='Yaw observado'); #predicted yaw
    # plt.xlabel('Tiempo')
    # plt.ylabel('Yaw')
    # plt.plot(tx,yaw_spline1,'r', label='Yaw esperado') #true yaw
    # plt.legend()
    return RMSEr, percent_error_xy

# bs1 = 'coordenadas_bspline_rmse_bs1.txt'
# bs10 = 'coordenadas_bspline_rmse_bs10.txt'

# RMSE_xy = calcula_error(bs1, bs10)