import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from controller import Supervisor
from collections import deque
import random
import math

Goal_completed = 0
# Create an instance of the supervisor
supervisor = Supervisor()

# Get the robot node
robot_node = supervisor.getFromDef('Thymio')
initial_translation_field = robot_node.getField('translation')
initial_translation = initial_translation_field.getSFVec3f()
initial_rotation_field = robot_node.getField('rotation')
initial_rotation = initial_rotation_field.getSFRotation()

time_step = int(supervisor.getBasicTimeStep())  # Changed from Robot to Supervisor



# Define the environment
action_space_size = 4  # left, right, forward, backward
state_space_size = 10  # 7 sensor readings + 2 GPS coordinates
goal_position = np.array([-0.665309, -0.643831, -0.0143212])

alpha = 0.2  # learning rate
gamma = 0.75  # discount factor
epsilon = 1.0  # increased exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999  # decay epsilon more slowly
training_episodes = 1000
batch_size = 64  # batch size for training the model
memory_size = 20000  # size of the memory for storing experiences

# Get the proximities and enable them
prox = [supervisor.getDevice(f'prox.horizontal.{i}') for i in range(7)]
for p in prox:
    p.enable(time_step)

# Get the GPS device and enable it
gps = supervisor.getDevice('gps')
gps.enable(time_step)

# Get the devices and enable them
left_motor = supervisor.getDevice('motor.left')
right_motor = supervisor.getDevice('motor.right')

# Set the motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Your existing code from here on is fine, I've just omitted it for brevity...

# Define the neural network for the Q function approximation
model = tf.keras.Sequential([
    layers.Dense(24, activation='relu', input_dim=state_space_size),
    layers.Dense(24, activation='relu'),
    layers.Dense(action_space_size, activation='linear')
])

# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))


# Initialize memory for storing experiences
memory = deque(maxlen=memory_size)

# Function to calculate the reward for a state
def calculate_reward(state, next_state, hit_wall_flag):  
    penalty_for_wall = -20
    penalty_for_time = -0.5  # Reduced penalty for time
    penalty_for_distance = -0.5
    reward_for_goal = 500
    reward_for_avoiding_wall = 2  # Reward for each time step without hitting a wall
    reward_for_closer_to_goal = 5  # Increased reward for getting closer to goal

    # calculate distance from the goal
    current_position = np.array([state[7], state[8], state[9]])
    next_position = np.array([next_state[0][7], next_state[0][8], next_state[0][9]])
    distance = np.linalg.norm(current_position - goal_position)

    reward = penalty_for_time + penalty_for_distance * math.log(distance + 1)
    if np.linalg.norm(next_position - goal_position) < np.linalg.norm(current_position - goal_position):
        reward += reward_for_closer_to_goal  # add bonus reward for moving closer to the goal
    if hit_wall_flag:
        reward += penalty_for_wall
    else:
        reward += reward_for_avoiding_wall  # add bonus reward for not hitting a wall
    if at_goal(next_position, goal_position):
        reward += reward_for_goal

    return reward


def move_robot(action):
    speed = 4
    if action == 0:  # move forward
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
    elif action == 1:  # move backward
        left_motor.setVelocity(-speed)
        right_motor.setVelocity(-speed)
    elif action == 2:  # turn left
        left_motor.setVelocity(-speed)
        right_motor.setVelocity(speed)
    elif action == 3:  # turn right
        left_motor.setVelocity(speed)
        right_motor.setVelocity(-speed)

def hit_wall(prox_values):
    return np.any(np.array(prox_values) > 2750)

def at_goal(gps_coordinates, target_location):
    return np.linalg.norm(gps_coordinates - target_location) < 0.07

# Loop for each episode
for i_episode in range(training_episodes):
    
    # Print the start of the episode
    print(f"Starting episode {i_episode + 1}/{training_episodes}")
    print(f"The robot completed the maze {Goal_completed} times")

    # Get the initial state
    state = [p.getValue() for p in prox] + list(gps.getValues())
    state = np.reshape(state, [1, state_space_size])
    
    # Loop for each step in the episode
    no_progress_threshold = 1000  # Number of steps without progress before ending the episode
    steps_without_progress = 0
    previous_distance = np.inf
    reset = False
        
    # Loop for each step in the episode
    for t in range(5000):

        if supervisor.step(time_step) == -1:
        
            # Reset robot's position and rotation
            #robot_node.getField('translation').setSFVec3f(initial_translation)
            #robot_node.getField('rotation').setSFRotation(initial_rotation)
            initial_translation_field.setSFVec3f(initial_translation)
            initial_rotation_field.setSFRotation(initial_rotation)
            #supervisor.simulationResetPhysics()  # Reset the physics of the simulation
            break

        # Choose an action
        action_values = model.predict(state)
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(action_space_size)  # Explore
        else:
            action = np.argmax(action_values[0])  # Exploit


               
        # Perform the action to get the next state, reward, and done
        move_robot(action)  # Execute the action
        next_state = [p.getValue() for p in prox] + list(gps.getValues())
        next_state = np.reshape(next_state, [1, state_space_size])
        done = at_goal(gps.getValues(), goal_position)
        reward = calculate_reward(state[0], next_state, hit_wall([p.getValue() for p in prox]))
    
    
        # Store the experience in memory
        memory.append((state, action, reward, next_state, done))

        # Update the state
        state = next_state
        
        # Print the state, action, and reward
        if t % 1000 == 0:  # Print every 10 steps
            print(f"Step: {t}, State: {state}, Action: {action}, Reward: {reward}")
        
        # Calculate the distance to the goal
        distance_to_goal = np.linalg.norm(next_state[0][7:10] - goal_position)
    
        # Check if the robot is making progress
        if distance_to_goal < previous_distance:
            previous_distance = distance_to_goal
            steps_without_progress = 0
        else:
            steps_without_progress += 1

        # If the robot hasn't made progress for a certain number of steps, end the episode
        if steps_without_progress >= no_progress_threshold:
            done = True
            reset = True

        # If done, break the loop 
        if done:
            if not reset:
                Goal_completed += 1
            #print(f"episode: {i_episode}/{training_episodes}, score: {t}, e: {epsilon:.2}")
            print(f"Finished episode {i_episode + 1}/{training_episodes}, Final score: {t}, e: {epsilon:.2}")
            
            # Reset robot's position and rotation
            #robot_node.getField('translation').setSFVec3f(initial_translation)
            #robot_node.getField('rotation').setSFRotation(initial_rotation)
            initial_translation_field.setSFVec3f(initial_translation)
            initial_rotation_field.setSFRotation(initial_rotation)
            #supervisor.simulationResetPhysics()  # Reset the physics of the simulation

            break


    # Train the model with a batch of experiences
    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target_f = model.predict(state)
            target = reward if done else reward + gamma * np.amax(target_f[0])
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

    

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Save the model
    model.save(f'my_model_{i_episode+1}')
     
    initial_translation_field.setSFVec3f(initial_translation)
    initial_rotation_field.setSFRotation(initial_rotation)
    #supervisor.simulationResetPhysics()  # Reset the physics of the simulation
