motor_example.cpp
	Testeo del movimiento individual de cada una de las helices
	 
motor_flight.cpp
	Intento de estabilizar el dron en el aire con velocidades de los rotores
	
topic_position.cpp
	Nodo que se subsrcirbe al topico position.point.z
	
rotors_rlx.cpp
	Nodo se subscribe a topico y recibe msg.x, luego publica velocidades a los rotores dependiendo de la pocision.
	 
rotors_rl.cpp
	Nodo se subscribe a topico y recibe msg.z, luego publica velocidades a los rotores dependiendo de la pocision.
	 
random_waypoint.cpp
	Es una copia de hovering_example.cpp solo se diferencia por un ciclo for y la funcion rand() 
	Nodo publicador de waypoints random, contiene servidor Gazebo paused/unpaused. 
	usarlo con esta linea en terminal ROS:
	roslaunch rotors_gazebo mav_random_waypoint.launch mav_name:=iris world_name:=basic	
	
waypoint_publisher(comentado).cpp
	copia de waypoint_publisher.cpp comentado linea por lina, solo con fines de estudio
	
test_map_wp.cpp
	copia de random_waypoint.cpp, con la modificacion de que ahora genera waypoints solo en el eje x hasta colicionar con una pared
	
reset_world.cpp
	Utiliza el servicio gazebo reset_world, usandolo en reset_world.lauch en conjunto con waypoint_publisher_file.cpp y waypoint_publisher.cpp restablece al dron a su poscision inicial
	Debe estar ejecutado mav_with_waypoint_publisher.launch para que lo anterior funcione, mas detallen en https://efficacious-gondola-c80.notion.site/Reset-UAV-Position-5bc467d6a6f34490ac7eaa1df943d9a6
