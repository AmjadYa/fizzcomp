#!/usr/bin/env python3

import sys
import rospy
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelState

class TeleportHandler:
    def __init__(self, model_name='B1'):
        self.model_name = model_name

        # Define teleport positions
        self.teleport_positions = {
            'TP1': {
                'position': {'x': 5.49942880783774, 'y': 2.504030700996579, 'z': 0.04000039797012949},
                'orientation': {'x': 7.377270669093508e-07, 'y': -4.3205541389025773e-07, 'z': -0.7090994033738673, 'w': 0.7051085279119056}
            },
            'TP2': {
                'position': {'x': 5.3474887443153785, 'y': -0.9349569584188745, 'z': 0.04000037723629943},
                'orientation': {'x': 1.7321540120358244e-06, 'y': 8.610931097589084e-07, 'z': -0.814923731390067, 'w': 0.5795682117003553}
            },
            'TP3': {
                'position': {'x': 4.31305060031579, 'y': -1.3878525178704242, 'z': 0.04000050138909643},
                'orientation': {'x': 2.8548229122584364e-07, 'y': 5.904484287748461e-08, 'z': -0.9992515317144463, 'w': -0.038683024264505914}
            },
            'TP4': {
                'position': {'x': 0.6491008198665422, 'y': -0.9207976055506509, 'z': 0.04000053773996677},
                'orientation': {'x': 5.673153474413703e-07, 'y': 1.1238297852020603e-07, 'z': -0.6684896009764731, 'w': -0.7437214891247805}
            },
            'TP5': {
                'position': {'x': 0.6689328884773375, 'y': 2.01851026363387, 'z': 0.04000094070292316},
                'orientation': {'x': -2.860153819036568e-06, 'y': 3.005876131366921e-06, 'z': 0.7085922122246099, 'w': -0.7056182230905165}
            },
            'TP6': {
                'position': {'x': -3.0218217082015695, 'y': 1.5572375439630923, 'z': 0.03997891270315111},
                'orientation': {'x': -4.068919485243806e-05, 'y': -9.436718605393764e-05, 'z': 0.9984824087289902, 'w': 0.05507148897549148}
            },
            'TP7': {
                'position': {'x': -4.301637175421005, 'y': -2.312349652507272, 'z': 0.03998563711874584},
                'orientation': {'x': -6.732302218612708e-06, 'y': 0.00012692216855302477, 'z': -0.05064239775258873, 'w': -0.9987168424510062}
            },
            'TP8': {
                'position': {'x': -1.2089846301463472, 'y': -1.186840844647, 'z': 1.8503400563324273},
                'orientation': {'x': -0.0057606116672731245, 'y': -0.013744051998107805, 'z': -0.0008294302102146539, 'w': -0.9998886080126218}
            }
        }

        # Set up the teleport service proxy
        rospy.loginfo("Waiting for /gazebo/set_model_state service...")
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.loginfo("/gazebo/set_model_state service is available.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service initialization failed: {e}")
            sys.exit(1)

    def teleport_to_position(self, tp_name):
        """
        Teleports the robot to the specified TP position.

        :param tp_name: String name of the TP button (e.g., 'TP1', 'TP2', ...)
        """
        if tp_name not in self.teleport_positions:
            rospy.logerr(f"Teleport position '{tp_name}' not defined.")
            return

        position = self.teleport_positions[tp_name]['position']
        orientation = self.teleport_positions[tp_name]['orientation']

        # Create a ModelState message
        model_state = ModelState()
        model_state.model_name = self.model_name
        model_state.pose.position.x = position['x']
        model_state.pose.position.y = position['y']
        model_state.pose.position.z = position['z']
        model_state.pose.orientation.x = orientation['x']
        model_state.pose.orientation.y = orientation['y']
        model_state.pose.orientation.z = orientation['z']
        model_state.pose.orientation.w = orientation['w']
        model_state.reference_frame = 'world'  # Relative to the 'world' frame

        # Create the service request
        set_state_request = SetModelStateRequest()
        set_state_request.model_state = model_state

        try:
            # Call the service to set the model state
            response = self.set_model_state_service(set_state_request)
            if response.success:
                rospy.loginfo(f"Successfully teleported to {tp_name}.")
            else:
                rospy.logerr(f"Failed to teleport to {tp_name}: {response.status_message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
