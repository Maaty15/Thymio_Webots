import random  # We import the random module at the beginning of the file
import math
from controller import Robot

# Create an instance of the robot
robot = Robot()

# Get the devices and enable them
left_motor = robot.getDevice('motor.left')
right_motor = robot.getDevice('motor.right')
prox_0 = robot.getDevice('prox.horizontal.0')
prox_1 = robot.getDevice('prox.horizontal.1')
prox_2 = robot.getDevice('prox.horizontal.2')
prox_3 = robot.getDevice('prox.horizontal.3')
prox_4 = robot.getDevice('prox.horizontal.4')
prox_5 = robot.getDevice('prox.horizontal.5')
prox_6 = robot.getDevice('prox.horizontal.6')

prox_0.enable(10)
prox_1.enable(10)
prox_2.enable(10)
prox_3.enable(10)
prox_4.enable(10)
prox_5.enable(10)
prox_6.enable(10)



# Set the motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Set the motor speeds
speed = 7
left_motor.setVelocity(speed)
right_motor.setVelocity(speed)

# Get the GPS device and enable it
gps = robot.getDevice('gps')
gps.enable(10)  # Enable the GPS. Here 10 is the sampling period of the sensor in milliseconds.

gps_coordinates = gps.getValues()
print(f"GPS: {gps_coordinates}")

# Read the proximity sensor values

prox_0_value = prox_0.getValue()
prox_1_value = prox_1.getValue()
prox_2_value = prox_2.getValue()
prox_3_value = prox_3.getValue()
prox_4_value = prox_4.getValue()
prox_5_value = prox_5.getValue()
prox_6_value = prox_6.getValue()

print(prox_0_value)
print(prox_1_value)
print(prox_2_value)
print(prox_3_value)
print(prox_4_value)
print(prox_5_value)

# Define the target location
target_location = [-0.909317, -0.393856, -0.0186191]  # Replace X, Y, and Z with the target coordinates

# Define the reward system
reward = 0
reward_per_step = 0.45  # The reward for each move without hitting a wall
penalty_for_collision = -1.5  # The penalty for hitting a wall
reward_for_goal = 500  # The reward for reaching the goal
print(reward)

scaling_factor = 0.25  # Define a scaling factor. This is an example, adjust it according to your needs.

# Define a time penalty
time_penalty = -0.2  # An example value; adjust as needed


#Calculating the distamce between the robot and the end goal
def calculate_distance(current, goal):
    """Calculate the Euclidean distance between current and goal."""
    return math.sqrt(sum((c - g) ** 2 for c, g in zip(current, goal)))

start_position = gps.getValues()  # Store the starting position of the robot

# Main control loop
while robot.step(10) != -1:

    # Get the GPS coordinates
    gps_coordinates = gps.getValues()
    print(f"GPS: {gps_coordinates}")
    
    # Read the proximity sensor values
    prox_0_value = prox_0.getValue()
    prox_1_value = prox_1.getValue()
    prox_2_value = prox_2.getValue()
    prox_3_value = prox_3.getValue()
    prox_4_value = prox_4.getValue()
    prox_5_value = prox_5.getValue()
    prox_6_value = prox_6.getValue()

    print(prox_0_value)
    print(prox_1_value)
    print(prox_2_value)
    print(prox_3_value)
    print(prox_4_value)
    print(prox_5_value)
    print(prox_6_value)
    
    # Calculate the distance to the goal
    distance = calculate_distance(gps_coordinates, target_location)
    print(f"Euclidean Distance : {distance}")
    
    # Define a distance reward (or penalty)
    # This could be a function of distance, e.g., negative if the robot is moving away from the goal
    distance_reward =  -distance * scaling_factor
    print(f"The Distance Reward : {distance_reward}")

    # Update the reward
    reward += distance_reward
    
    # Apply the time penalty
    reward += time_penalty
    print(f"Time Penalty: {time_penalty}")
    

     # Check if the GPS coordinates match the target location
    if (abs(gps_coordinates[0] - target_location[0]) < 0.07 and  # 0.1 is the threshold, adjust it according to your needs
        abs(gps_coordinates[1] - target_location[1]) < 0.07 and
        abs(gps_coordinates[2] - target_location[2]) < 0.07):

        # The robot has reached the target location
        print("End Goal Reached")
        
        # Add the goal reward
        reward += reward_for_goal
        print(f"Reward: {reward}")
        
        # Stop the robot
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        
        break
        
    # Check if there is a wall in front
    if prox_0_value > 2750 or prox_1_value > 2750 or prox_2_value > 2750 or prox_3_value > 2750 or prox_4_value > 2750 or prox_5_value > 2750 or prox_6_value > 2750:
        
        # Stop the robot
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        

        
        
        print(f"Collision! Reward: {reward}")

        print("Wall detected!")


        # Decide a random direction to turn: left or right
        random_direction = random.choice(['left', 'right'])

        if random_direction == 'left':
            # Turn the robot to the left
            left_motor.setVelocity(-speed)
            right_motor.setVelocity(speed)
        else:
            # Turn the robot to the right
            left_motor.setVelocity(speed)
            right_motor.setVelocity(-speed)

        # Print the direction the robot will turn
        print(f"Turning {random_direction}")

        # Wait for the robot to turn
        robot.step(500)
    else:
        # No wall detected, continue moving forward
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        
        # Apply the step reward
        reward += reward_per_step
        print(f"Reward: {reward}")