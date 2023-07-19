from controller import Supervisor

# create the Supervisor instance.
supervisor = Supervisor()

# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())

# Get device and enable it
left_motor = supervisor.getDevice('motor.left')
right_motor = supervisor.getDevice('motor.right')

# Get the robot's translation and rotation fields
robot_node = supervisor.getFromDef('Thymio')
initial_translation_field = robot_node.getField('translation')
initial_translation = initial_translation_field.getSFVec3f()
initial_rotation_field = robot_node.getField('rotation')
initial_rotation = initial_rotation_field.getSFRotation()

prox_0 = supervisor.getDevice('prox.horizontal.0')
prox_1 = supervisor.getDevice('prox.horizontal.1')
prox_2 = supervisor.getDevice('prox.horizontal.2')
prox_3 = supervisor.getDevice('prox.horizontal.3')
prox_4 = supervisor.getDevice('prox.horizontal.4')
prox_5 = supervisor.getDevice('prox.horizontal.5')
prox_6 = supervisor.getDevice('prox.horizontal.6')

prox_0.enable(timestep)
prox_1.enable(timestep)
prox_2.enable(timestep)
prox_3.enable(timestep)
prox_4.enable(timestep)
prox_5.enable(timestep)
prox_6.enable(timestep)

# Set the motors to velocity control mode
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

while supervisor.step(timestep) != -1:
    prox_0_value = prox_0.getValue()
    prox_1_value = prox_1.getValue()
    prox_2_value = prox_2.getValue()
    prox_3_value = prox_3.getValue()
    prox_4_value = prox_4.getValue()
    prox_5_value = prox_5.getValue()
    prox_6_value = prox_6.getValue()
    
    # If the robot hits the wall, reset its position and rotation
    if prox_0_value > 2750 or prox_1_value > 2750 or prox_2_value > 2750 or prox_3_value > 2750 or prox_4_value > 2750 or prox_5_value > 2750 or prox_6_value > 2750:
        initial_translation_field.setSFVec3f(initial_translation)
        initial_rotation_field.setSFRotation(initial_rotation)
        supervisor.simulationResetPhysics()  # Reset the physics of the simulation
    else:
        # Set velocity to move forward
        left_motor.setVelocity(5.0)
        right_motor.setVelocity(5.0)
