#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

rospy.init_node('topic_publisher')
pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

move = Twist()
move.linear.x = 0.5
move.angular.z = 0.0

rospy.sleep(1)
pub_score.publish(String('sherlock,detective,0,AAAA'))
pub_cmd.publish(move)
rospy.sleep(2)
pub_score.publish(String('sherlock,detective,-1,AAAA'))