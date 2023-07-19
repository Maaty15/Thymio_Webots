import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from controller import Robot, Motor, DistanceSensor, GPS, Supervisor

class DQNAgent:
    def __init__(self, input_size, action_size):
        self.input_size = input_size
        self.action_size = action_size

        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.input_size, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam())

        self.gamma = 0.95  # discount factor for future rewards
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995  # decay rate for exploration
        self.epsilon_min = 0.01  # minimum exploration rate

        self.memory = []  # memory for experience replay

    def select_action(self, state):
        # epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        # store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # experience replay
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self, reward, done):
        if len(self.memory) > 32:
            self.replay(32)


class ThymioMazeNavigator:
    def __init__(self):
        # initialize the Robot superclass
        self.robot = Robot()
        
        # set the timestep
        self.time_step = 64
        
        # initialize the GPS
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.time_step)
        
        # initialize the Supervisor
        self.supervisor_instance = Supervisor()  
        
        # initialize sensors
        self.front_sensor = self.robot.getDistanceSensor('prox.horizontal.4')
        self.front_sensor.enable(self.time_step)
        
        # initialize motors
        self.left_motor = self.robot.getMotor('motor.left')
        self.right_motor = self.robot.getMotor('motor.right')
        self.left_motor.setPosition(float('inf'))  # to set the motors to velocity control mode
        self.right_motor.setPosition(float('inf'))  # to set the motors to velocity control mode
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # goal position
        self.goal_position = [-0.929412, -0.373856, -0.0181303]
        
        # initialize the DQN agent
        self.rl_agent = DQNAgent(input_size=4, action_size=3)  # change input_size to match the size of the observations


    def get_observation(self):
        # get the values from the sensors
        front_sensor_value = self.front_sensor.getValue()
        
        # get the robot's position
        position = self.gps.getValues()
        
        return [front_sensor_value, position[0], position[1], position[2]]  # assuming the position is a 3D coordinate (x, y, z)
    
    def is_off_track(self, current_position):
        # Define the boundaries of the track
        x_min = -1.0
        x_max = 1.0
        y_min = -1.0
        y_max = 1.0
    
        # Check if the robot's current position is within these boundaries
        if current_position[0] < x_min or current_position[0] > x_max or current_position[1] < y_min or current_position[1] > y_max:
            return True
        else:
            return False

    
    def get_reward(self):
        reward = 0
    
        # get the current position of the Thymio
        current_position = self.gps.getValues()
    
        # calculate the Euclidean distance to the goal
        distance_to_goal = np.linalg.norm(np.array(self.goal_position) - np.array(current_position))
    
        # if Thymio is off track, give a negative reward
        if self.is_off_track(current_position):
            reward = -100
        else:
            # if the Thymio is on track, give a reward based on how close it is to the goal
            reward = -distance_to_goal  # the closer to the goal, the higher the reward
    
        # if Thymio reached the goal, give a big reward
        if self.is_done():
            reward = 500
    
        return reward
    

    def is_done(self, observation):
        position = self.gps.getValues()

        # Check if the robot has reached the goal
        distance_to_goal = np.linalg.norm(np.array(position) - np.array(self.goal_position))

        # The episode is done if the robot has reached the goal
        return distance_to_goal < 0.125
    

    def set_action(self, action):
        # map the action to motor commands
        if action == 0:  # go left
            left_speed = -9.0
            right_speed = 9.0
        elif action == 1:  # go right
            left_speed = 9.0
            right_speed = -9.0
        elif action == 2:  # go straight
            left_speed = 9.0
            right_speed = 9.0
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def run(self, num_episodes=1000):
        for episode in range(num_episodes):
            self.supervisor_instance.simulationReset()  # reset the simulation
            while self.robot.step(self.time_step) != -1:
                observation = self.get_observation()
                action = self.rl_agent.select_action(observation)
                self.set_action(action)
                reward = self.get_reward()
                done = self.is_done(observation)
                self.rl_agent.remember(observation, action, reward, self.get_observation(), done)
                self.rl_agent.update(reward, done)
                if done:
                    break
