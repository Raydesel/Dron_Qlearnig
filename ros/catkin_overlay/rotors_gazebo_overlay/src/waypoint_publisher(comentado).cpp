/*Autor: Raydesel 
Fecha: 30/sep/2022
*/

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <mav_msgs/conversions.h>
#include <mav_msgs/default_topics.h>
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

//Funcion para generar un vector con las cordenadas [x,y,z,(yaw)] 
void waypoint_generator(geometry_msgs::PointStamped msg){
    
}
//cadenas argc y argv en este nodo no son usadas, solo se definen por convencion
int main(int argc, char** argv){
    //Inicializacion del nodo waypoint_publisher
    ros::init(argc, argv, "waypoint_publisher"); 
    //Inicializacion de NodeHandle para comunciacion con ROS system
    ros::NodeHandle nh;
    //Se anuncia el publicador para el topico command_trajectory
    ros::Publisher trajectory_pub =
        nh.advertise<trajectory_msg::MultiDOFJointTrajectory>(
            mav_msgs::default_topics::COMMAND_TRAJECTORY, 10);
    //imprime en consola msg
    ROS_INFO("Started waypoint_publisher.");
    //variable tipo double para definir un delay
    double delay = 1.0;
    //La constante M_PI, esta predefinida 
    const float DEG_2_RAD = M_PI / 180.0;

    //define un mensaje del tipo trajectory_msg Multi... 
    trajectory_msg::MultiDOFJointTrajectory trajectory_msg;
    //escribe en el mensaje el tiempo ros
    trajectory_msg.header.stamp = ros::Time::now();

    //Extrae cordenadas [x,y,z] y lo transforma en un Eigenvector3d
    Eigen::Vector3d desired_position(std::stof(args.at(1)), std::stof(args.at(2)),
                                     std::stof(args.at(3)));
    //Extrae angulo yaw y lo transforma a radianes
    double desired_yaw = std::stof(args.at(4)) * DEG_2_RAD;
    
    //Define un mensage que contiene la informacion de time, cordenadas y yaw
    mav_msgs::msgMultiDOFJointTrajectoryFromPositionYaw(desired_position,
        desired_yaw, &trajectory_msg);

    //Asigna un delar a ROS.
    ros::Duration(delay).sleep();
    //Pregunta al publicador numero de subs
    while (trajectory_pub.getNumSubscribers() == 0 && ros::ok()) {
        ROS_INFO("No existe una subscripcion disponible, intentado otra vez en 1 segundo.");
        ros::Duration(1.0).sleep();
    }
    //Imprime en consola
    ROS_INFO("Publicando waypoint en el espacio %s: [%f, %f, %f].",
            nh.getNamespace().c_str(),
            desired_position.x(),
            desired_position.y(),
            desired_position.z());
    //Publica trajectory mensaje
    trajectory_pub.publish(trajectory_msg);

    ros::spinOnce();
    ros::shutdown();

    return 0;
} 