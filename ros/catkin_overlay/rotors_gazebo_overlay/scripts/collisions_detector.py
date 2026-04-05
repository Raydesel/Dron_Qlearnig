#!/usr/bin/env python

import rospy
# Librerias para leer mensajes de posicion en z 
# del topico /iris/odometry_sensor1/pose
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool

def pose_callback(msg):
    # Obtener la posicion en el eje z de la pose recibida
    z_position = msg.position.z
    print(z_position)
    rospy.loginfo(z_position)

    # Verificar si la altura es menor a 1.5
    if z_position < 1.5:
        # Publicar un mensaje de colision en el topico '/collision'
        print('collision is detected')

def subscriber_pose():
    rospy.init_node('collision_detection_node', anonymous=True)
    # Suscribirse al topico '/iris/odometry_sensor1/pose'
    rospy.Subscriber('/iris/odometry_sensor1/pose', Pose, pose_callback)  
    rospy.spin()

if __name__ == '__main__':
    subscriber_pose()
