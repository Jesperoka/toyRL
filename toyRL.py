import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# TODOLIST
# ============================================
# TODO: use explicit data types for all arrays
# TODO pass physics params into class constructor
# TODO enforce positive drag coefficient (saturate delta output)

# ============================================


# =========================
# Robot and Dart Simulation
# =========================

class RobotArm:
    def __init__(self, link_lengths):
        self.link_lengths = link_lengths
        self.joint_angles = [0] * len(link_lengths)
    
    def forward_kinematics(self):
        x, y = 0.0, 0.0
        theta = 0.0
        
        points = [(x, y)]
        for i in range(len(self.joint_angles)):
            theta += self.joint_angles[i]
            x += self.link_lengths[i] * np.cos(theta)
            y += self.link_lengths[i] * np.sin(theta)
            points.append((x, y))
        return points

class Dart:
    def __init__(self):
        self.baseline_drag_coefficient = 0.1
        self.baseline_initial_velocity = np.array([30.0, 20.0])
        self.initial_position = np.array([0.0, 0.0])

        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.delta_drag_coefficient = 0.0
        self.delta_initial_velocity = np.array([0.0, 0.0]) 

        self.gravity = np.array([0, -9.81])  # Gravity force, m/s^2 downward

    def launch(self):
        self.velocity = self.baseline_initial_velocity + self.delta_initial_velocity 
   
    def update(self, dt=0.01):
        # Update velocity due to drag
        drag_force = 0.5 * (self.baseline_drag_coefficient + self.delta_drag_coefficient) * np.linalg.norm(self.velocity) * self.velocity
        self.velocity -= drag_force * dt
        
        # Update velocity due to gravity
        self.velocity += self.gravity * dt
        
        # Update position
        self.position += self.velocity * dt

    def reset(self, initial_position=None, initial_velocity=None):
        """Reset the Dart's position and velocity."""
        if initial_position is None: initial_position = self.initial_position.copy()
        if initial_velocity is None: initial_velocity = self.baseline_initial_velocity.copy()
        self.position = initial_position
        self.velocity = initial_velocity

    def is_in_playing_field(self):
        return 100 >= self.position[0] and self.position[0] >= -100.0 and 100 >= self.position[1] and self.position[1] >= -100.0

# =====================
# Visualization Methods
# =====================

def animate(i, arm, dart_positions, line):
    # Update dart position for this frame
    line.set_data(dart_positions[i][0], dart_positions[i][1])
    return line,

def visualize(robot_arm, dart_positions, target_position):
    arm_points = robot_arm.forward_kinematics()
    arm_x, arm_y = zip(*arm_points)
    fig, ax = plt.subplots()
    
    # Draw Robot Arm
    ax.plot(arm_x, arm_y, '-o', label='Robot Arm')
    
    # Draw Target
    ax.scatter(*target_position, c='green', marker='x', label='Target')
    
    # Set limits and legend
    # ax.set_xlim(-sum(robot_arm.link_lengths), sum(robot_arm.link_lengths))
    # ax.set_ylim(-sum(robot_arm.link_lengths), sum(robot_arm.link_lengths))
    ax.legend()
    
    # Initialize dart's trajectory line
    line, = ax.plot([], [], 'ro', label='Dart')

    # Create animation
    ani = FuncAnimation(fig, animate, frames=len(dart_positions), fargs=(robot_arm, dart_positions, line), 
                        interval=50, blit=True)
    
    plt.show()


# ============================
# Policy and REINFORCE Methods
# ============================

def neural_network(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    b2 = np.zeros((1, output_dim))
    return [W1, b1, W2, b2]

def policy_forward(x, model):
    W1, b1, W2, b2 = model
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    if np.isnan(z2).any(): 
        print("gotnans", x)
        print(W1)
        print(W2)
        print(b1)
        print(b2)
        input()
    return z2, a1

def policy_backward(x, eph, epdlogp, model, clip_norm = 1.0):
    W1, b1, W2, b2 = model

    # Output layer gradients
    dW2 = np.dot(eph.T, epdlogp)
    db2 = np.sum(epdlogp, axis=0, keepdims=True)
    
    # Hidden layer gradient
    dhidden = np.dot(epdlogp, W2.T) * (1 - eph**2)
    
    # Gradients for W1 and b1
    dW1 = np.dot(x.T, dhidden)  # Assumes X is your batch of input data
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    # Compute L2 norm of all gradients
    grad_norm = np.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2))
    
    # If norm exceeds clip_norm, scale all gradients
    if np.isnan(grad_norm).any(): 
        print("\n\n\nuh oh spaghettio\n\n\n")
        print("x")
        print(x)
        print("eph")
        print(eph)
        print("epdlogp")
        print(epdlogp)
        print("model")
        print(model)
        print("dW1")
        print(dW1)
        print("dW2")
        print(dW2)
        print("db1")

        print(db1)
        print("db2")
        print(db2)

    if grad_norm > clip_norm:
        scale_factor = clip_norm / grad_norm
        dW1 *= scale_factor
        db1 *= scale_factor
        dW2 *= scale_factor
        db2 *= scale_factor

    return [dW1, db1, dW2, db2]

def sample_action(mean, std_dev):
    return mean + std_dev * np.random.randn()

def sample_trajectory(arm, dart, policy_model, batch_size, predicted_trajectory, target_position, max_time_steps=100):
    eph_list = []
    actions = []
    rewards = []
    dart_trajectories = []
    predicted_trajectory = np.vstack(predicted_trajectory).flatten()

    # Throw batch size number of darts
    for i in range(batch_size):
        dart.reset()
        went_far_enough = False
        dart_trajectory = []

        # Take an action
        action_mean, eph = policy_forward(predicted_trajectory, policy_model)
        action = sample_action(action_mean, 0.1)
        actions.append(action)
        eph_list.append(eph)

        # Set the Dart's initial velocity and drag coefficient deltas
        dart.delta_drag_coefficient = action[0, 2]      
        dart.delta_initial_velocity = action[0, 0:2]    
        dart.launch()

        # Simulate dart trajectory
        for _ in range(max_time_steps):
            if dart.is_in_playing_field(): 
                dart.update()
            
            dart_trajectory.append(dart.position.copy())
           
            # When/if dart goes beyond target, calculate reward based on distance 
            if dart.position[0] >= target_position[0] and not went_far_enough:  
                distance_to_target = np.linalg.norm(dart.position - target_position)
                rewards.append(-distance_to_target)
                went_far_enough = True
        
        # Otherwise calculate based on where it ended up 
        if not went_far_enough:
            distance_to_target = np.linalg.norm(dart.position - target_position)
            rewards.append(-distance_to_target)

        dart_trajectories.append(np.vstack(dart_trajectory).flatten())

    return dart_trajectories, actions, rewards, eph_list



def reinforce(arm, dart, policy_model, batch_size, predicted_trajectory, target_position, learning_rate=1e-2, gamma=0.99):
    dart_trajectories, actions, rewards, eph_list = sample_trajectory(arm, dart, policy_model, batch_size, predicted_trajectory, target_position)

    discounted_rewards = np.zeros_like(rewards)
    running_add = 0.0

    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    epdlogp = np.vstack(actions)
    epdlogp *= discounted_rewards[:, np.newaxis]

    grads = policy_backward(np.vstack(dart_trajectories), np.vstack(eph_list), epdlogp, policy_model)

    for i, grad in enumerate(grads):
        policy_model[i] += learning_rate * grad

    return policy_model

def true_trajectory(dart, initial_velocity, max_time_steps=100, dt=0.01):
    dart_positions = []

    dart.baseline_initial_velocity = initial_velocity
    dart.launch()
    
    for _ in range(max_time_steps):
        dart_positions.append(dart.position.copy())
        dart.update(dt)
    
    return dart_positions

def unflatten_to_pairs(flattened_array):
    if len(flattened_array) % 2 != 0:
        raise ValueError("Flattened array length must be even to be reshaped into pairs.")
    reshaped_array = flattened_array.reshape(-1, 2)
    list_of_arrays = [row for row in reshaped_array]
    return list_of_arrays
    

# =================
# Main Running Code
# =================

if __name__ == "__main__":
    arm = RobotArm([1, 1, 1])
    dart = Dart()
    true_dart = Dart()

    initial_velocity = np.array([20.0, 10.0])  # example value, adjust as needed
    target_position = np.array([10.0, 3.0])  # example value, adjust as needed

    true_traj = true_trajectory(true_dart, initial_velocity)

    policy_model = neural_network(input_dim=2*len(true_traj), hidden_dim=10, output_dim=3)

    batch_size = 10
    num_iterations = 10000
    for i in range(num_iterations):
        policy_model = reinforce(arm, dart, policy_model, batch_size, true_traj, target_position)
        
        # Visualize after some iterations
        if i+1 % 1000 == 0:
            action_mean, _ = policy_forward(np.vstack(true_traj).flatten(), policy_model)
            print(action_mean)
            dart_trajectories, _, _, _ = sample_trajectory(arm, dart, policy_model, 1, true_traj, target_position)
            visualize(arm, true_traj, target_position)
            visualize(arm, unflatten_to_pairs(dart_trajectories[0]), target_position)
