# bismillah_sequence.py

import rospy
from std_msgs.msg import String
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, QTimer
import time

class BismillahSequence(QObject):
    # Define signals to communicate with the main GUI thread
    set_linear = pyqtSignal(float)         # Emitted to set linear speed
    set_angular = pyqtSignal(float)        # Emitted to set angular speed
    stop_motion_signal = pyqtSignal()      # Emitted to stop motion
    request_predict = pyqtSignal()         # Emitted to request prediction
    teleport_signal = pyqtSignal(dict)      # Emitted to request teleportation

    def __init__(self, teleport_handler):
        super(BismillahSequence, self).__init__()
        self.teleport_handler = teleport_handler
        self._is_running = False
        self.current_step = 0

        # Initialize ROS Publisher for /score_tracker
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=10)
        rospy.sleep(0.5)  # Allow time for the publisher to register

        # Connect the prediction result slot
        self.prediction_result = None
        self.prediction_received = False
        self.prediction_text = None

    @pyqtSlot(str)
    def receive_prediction_result(self, prediction):
        rospy.loginfo(f"BismillahSequence received prediction: {prediction}")
        self.prediction_text = prediction
        self.prediction_received = True
        self.execute_next_step()

    @pyqtSlot()
    def run_sequence(self):
        if self._is_running:
            rospy.logwarn("Bismillah sequence is already running.")
            return
        self._is_running = True
        self.current_step = 0
        self.execute_next_step()

    def execute_next_step(self):
        if not self._is_running:
            return

        self.current_step += 1
        rospy.loginfo(f"Executing Step {self.current_step}")
        
        if self.current_step == 1:
            # Start Timer Before Stage 1
            start_message = 'sherlock,detective,0,AAAA'
            self.pub_score.publish(String(start_message))
            rospy.loginfo(f"Published start message: {start_message}")
            rospy.loginfo("Timer started.")
            QTimer.singleShot(3000, self.execute_next_step)  # Wait 3 seconds

        elif self.current_step == 2:
            # Stage 1: Move Forward
            self.set_linear.emit(2.7)
            self.set_angular.emit(0.0)
            rospy.loginfo("Stage 1: Move forward.")
            QTimer.singleShot(500, self.stop_motion)

        elif self.current_step == 3:
            # Stage 1: Motion stopped
            rospy.loginfo("Stage 1: Motion stopped.")
            # Stage 2: Turn Left
            self.set_linear.emit(0.0)
            self.set_angular.emit(2.0)
            rospy.loginfo("Stage 2: Turn left.")
            QTimer.singleShot(1000, self.stop_motion)

        elif self.current_step == 4:
            # Stage 2: Motion stopped
            rospy.loginfo("Stage 2: Motion stopped.")
            # Step 3: Predict
            self.request_predict.emit()
            rospy.loginfo("Step 3: Predict.")
            # Wait for prediction
            self.prediction_received = False

        elif self.current_step == 5:
            # After prediction
            if self.prediction_received:
                first_board = f'RAH,nuh_uh,1,{self.prediction_text}'
            else:
                first_board = 'RAH,nuh_uh,1,Prediction Failed'
            self.pub_score.publish(String(first_board))
            rospy.loginfo(f"Published first_board: {first_board}")
            # Step 4: Turn Right
            self.set_linear.emit(0.0)
            self.set_angular.emit(-2.1)
            rospy.loginfo("Step 4: Turn right.")
            QTimer.singleShot(920, self.stop_motion)

        elif self.current_step == 6:
            rospy.loginfo("Step 4: Motion stopped.")
            # Step 5: Move Forward
            self.set_linear.emit(3.0)
            self.set_angular.emit(0.0)
            rospy.loginfo("Step 5: Move forward.")
            QTimer.singleShot(2600, self.stop_motion)

        elif self.current_step == 7:
            rospy.loginfo("Step 5: Motion stopped.")
            # Step 6: Nudge Right
            self.set_linear.emit(0.0)
            self.set_angular.emit(-2.1)
            rospy.loginfo("Step 6: Nudge right.")
            QTimer.singleShot(600, self.stop_motion)

        elif self.current_step == 8:
            rospy.loginfo("Step 6: Motion stopped.")
            # Step 7: Predict
            self.request_predict.emit()
            rospy.loginfo("Step 7: Predict.")
            self.prediction_received = False

        elif self.current_step == 9:
            # After prediction
            if self.prediction_received:
                second_board = f'RAH,nuh_uh,2,{self.prediction_text}'
            else:
                second_board = 'RAH,nuh_uh,2,Prediction Failed'
            self.pub_score.publish(String(second_board))
            rospy.loginfo(f"Published second_board: {second_board}")
            # Step 8: Teleport to Position 1
            teleport_pose1 = {
                'position': {'x': 0.5527487156990568, 'y': -0.12600472106246535, 'z': 0.04000056025622947},
                'orientation': {'x': 2.448884472570069e-07, 'y': 1.278603394462025e-07, 'z': -0.7071178631556455, 'w': -0.7070956990437133}
            }
            self.teleport_signal.emit(teleport_pose1)
            rospy.loginfo("Step 8: Teleport to position 1.")
            QTimer.singleShot(2000, self.execute_next_step)

        # Continue similarly for other steps...
        elif self.current_step == 10:
            rospy.loginfo("Successfully teleported to the specified pose.")
            # Step 9: Move Backward
            self.set_linear.emit(-3.0)
            self.set_angular.emit(0.0)
            rospy.loginfo("Step 9: Move backward.")
            QTimer.singleShot(700, self.stop_motion)

        elif self.current_step == 11:
            rospy.loginfo("Step 9: Motion stopped.")
            # Step 10: Turn Right
            self.set_linear.emit(0.0)
            self.set_angular.emit(-2.0)
            rospy.loginfo("Step 10: Turn right.")
            QTimer.singleShot(700, self.stop_motion)

        elif self.current_step == 12:
            rospy.loginfo("Step 10: Motion stopped.")
            # Step 11: Predict
            self.request_predict.emit()
            rospy.loginfo("Step 11: Predict.")
            self.prediction_received = False

        elif self.current_step == 13:
            # After prediction
            if self.prediction_received:
                fourth_board = f'RAH,nuh_uh,4,{self.prediction_text}'
            else:
                fourth_board = 'RAH,nuh_uh,4,Prediction Failed'
            self.pub_score.publish(String(fourth_board))
            rospy.loginfo(f"Published fourth_board: {fourth_board}")
            # Step 12: Teleport to Position 2
            teleport_pose2 = {
                'position': {'x': -4.008629252948557, 'y': 0.4038460916553116, 'z': 0.04000060474749287},
                'orientation': {'x': 7.154773655035404e-07, 'y': -1.501085552816514e-06, 'z': -0.4271649858420221, 'w': -0.9041736972882035}
            }
            self.teleport_signal.emit(teleport_pose2)
            rospy.loginfo("Step 12: Teleport to position 2.")
            QTimer.singleShot(2000, self.execute_next_step)

        elif self.current_step == 14:
            rospy.loginfo("Successfully teleported to the specified pose.")
            # Step 13: Move Forward
            self.set_linear.emit(3.0)
            self.set_angular.emit(0.0)
            rospy.loginfo("Step 13: Move forward.")
            QTimer.singleShot(1300, self.stop_motion)

        elif self.current_step == 15:
            rospy.loginfo("Step 13: Motion stopped.")
            # Step 14: Turn Left
            self.set_linear.emit(0.0)
            self.set_angular.emit(1.8)
            rospy.loginfo("Step 14: Turn left.")
            QTimer.singleShot(2300, self.stop_motion)

        elif self.current_step == 16:
            rospy.loginfo("Step 14: Motion stopped.")
            # Step 15: Predict
            self.request_predict.emit()
            rospy.loginfo("Step 15: Predict.")
            self.prediction_received = False

        elif self.current_step == 17:
            # After prediction
            if self.prediction_received:
                sixth_board = f'RAH,nuh_uh,6,{self.prediction_text}'
            else:
                sixth_board = 'RAH,nuh_uh,6,Prediction Failed'
            self.pub_score.publish(String(sixth_board))
            rospy.loginfo(f"Published sixth_board: {sixth_board}")
            # Step 16: Teleport to Final Position
            teleport_pose3 = {
                'position': {'x': -4.048969918538875, 'y': -2.1927999150558106, 'z': 0.04000033111230297},
                'orientation': {'x': -7.721197054941669e-07, 'y': 4.2613242121936884e-07, 'z': -0.17655821924694795, 'w': -0.9842901986790121}
            }
            self.teleport_signal.emit(teleport_pose3)
            rospy.loginfo("Step 16: Teleport to final position.")
            QTimer.singleShot(2000, self.execute_next_step)

        elif self.current_step == 18:
            rospy.loginfo("Successfully teleported to the specified pose.")
            # Step 17: Predict
            self.request_predict.emit()
            rospy.loginfo("Step 17: Predict.")
            self.prediction_received = False

        elif self.current_step == 19:
            # After final prediction
            if self.prediction_received:
                seventh_board = f'RAH,nuh_uh,7,{self.prediction_text}'
            else:
                seventh_board = 'RAH,nuh_uh,7,Prediction Failed'
            self.pub_score.publish(String(seventh_board))
            rospy.loginfo(f"Published seventh_board: {seventh_board}")
            rospy.loginfo("Bismillah sequence completed.")

            # End Timer
            #end_time = time.time()
            #elapsed_time = end_time - start_time  # You'll need to define start_time appropriately
            #rospy.loginfo(f"Timer stopped. Elapsed time for the entire sequence: {elapsed_time:.2f} seconds.")

            # Publish stop message
            stop_message = 'sherlock,detective,-1,AAAA'
            self.pub_score.publish(String(stop_message))
            rospy.loginfo(f"Published stop message: {stop_message}")
            QTimer.singleShot(3000, self.finish_sequence)

    @pyqtSlot()
    def stop_motion(self):
        self.stop_motion_signal.emit()
        rospy.loginfo("Motion stopped.")
        self.execute_next_step()

    @pyqtSlot()
    def finish_sequence(self):
        self._is_running = False
        rospy.loginfo("Bismillah sequence fully completed.")
