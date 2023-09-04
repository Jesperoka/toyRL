import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from matplotlib.patches import FancyArrowPatch


# Second implementation
DTYPE = np.float64

class Dart:
    # Convenience
    pos = np.array([0, 0], DTYPE)
    vel = np.array([0, 0], DTYPE)
    angle = 0.0 
    length = 0.5

    # Parameters:
    # mass, gravity, air density, cross-section, drag, lift, inertia, 
    # CoM to fin aerodyn. center, CoM to drag aerodyn. center, area of each fin
    params = np.array([0.02, 9.81, 1.2, 0.0001, 0.08, 0.001, 0.0001, 0.01, 0.01, 0.01, 0.01], DTYPE)

    # State vector
    x = np.array([0, 0, 0, 0, 0, 0], DTYPE) # x, y, vx, vy, phi, wx

    def __init__(self) -> None:
        pass
    
    def dart_flight_2d(t, state, params):
        # Unpack state vector and parameters
        x, y, vx, vy, phi, wx = state
        m, g, rho, A, Cd_base, Cl_base, I, L, d, S, damping_coef = params

        # Calculate the angle of attack (difference between velocity direction and dart orientation)
        vel_angle = np.arctan2(vy, vx)
        #print(vx, vy, np.arctan2(vy, vx), phi)
        aoa = phi - vel_angle

        # Functions for Cl and Cd based on AoA
        def Cl_function(alpha):
            # Parameters
            Cl_max = 0.2
            alpha_stall = 15  # degrees
            
            if abs(alpha) < alpha_stall:
                Cl = 2 * Cl_max / alpha_stall * alpha
            else:
                Cl = -0.01 * (alpha - alpha_stall) + Cl_max  # drop after stall (simplified)
            
            return np.clip(Cl, -0.5, 0.5)  # Arbitrary bounds; adjust as needed

        def Cd_function(alpha):
            # Parameters
            Cd_min = 0.01
            alpha_stall = 15  # degrees
            Cd_stall = 0.2  # arbitrary value at stall
            
            if abs(alpha) < alpha_stall:
                Cd = Cd_min + 0.01 * abs(alpha)  # increase with AoA
            else:
                Cd = Cd_stall + 0.02 * (abs(alpha) - alpha_stall)  # increase more rapidly after stall
            
            return np.clip(Cd, 0.01, 1)  # Arbitrary bounds; adjust as needed


        Cd = Cd_function(aoa)
        Cl = Cl_function(aoa)

        # Calculate aerodynamic forces
        V = np.sqrt(vx**2 + vy**2)
        q = 0.5 * rho * V**2
        Fd_mag = -Cd * A * q 
        Fd = Fd_mag * np.array([np.cos(vel_angle), np.sin(vel_angle)], DTYPE)
        Fl_mag = Cl * A * q
        Fl = Fl_mag * np.array([-np.sin(vel_angle), np.cos(vel_angle)], DTYPE)

        Fg = np.array([0, -m * g], DTYPE)
        F = Fd + Fl + Fg

        # Moment due to lift
        M_lift = L * Fl[1]
        print(M_lift)

        # Moment due to drag
        M_drag = d * Fd_mag * np.sin(aoa)  # The sin(aoa) ensures we only get a moment if there's an angle between the dart and its direction of motion

        # Damping moment due to fletching
        damping_torque = -damping_coef * wx

        # Total moment
        M_total = M_lift + M_drag + damping_torque

        # Calculate state derivatives
        dxdt = vx
        dydt = vy
        dvxdt = F[0] / m
        dvydt = F[1] / m
        dphidt = wx
        dwxdt = M_total / I

        # Pack state derivatives into array
        state_dot = np.array([dxdt, dydt, dvxdt, dvydt, dphidt, dwxdt], DTYPE)

        return state_dot



class RobotArm:
    link1 = 1.0
    link2 = 1.0
    joint0_pos = np.array([0, 0])
    joint1_pos = np.array([-link1, 0], DTYPE)
    joint2_pos = np.array([-(link1+link2), 0], DTYPE)
    joint_angles = np.array([np.pi, np.pi], DTYPE)
    joint_angular_vels = np.array([0, 0], DTYPE)
    gripper_pos = np.array([0, link1+link2], DTYPE)
    gripper_relative_angle = 0.0 # angle relative to link2 
    release_angles = np.array([0, 0, 0], DTYPE) # angles of the joints at point of release 
    gripping = True

    def __init__(self) -> None:
        pass

    def move_arm(self, dt, dart: Dart=None, angular_vels=None, grip_rel_ang=None):
        # Optionally set angular velocities
        if angular_vels is not None: self.joint_angular_vels = angular_vels
        if grip_rel_ang is not None: self.gripper_relative_angle = grip_rel_ang

        # Update the joint angles
        self.joint_angles += self.joint_angular_vels * dt

        # Compute joint positions based on the current joint angles
        # First joint position remains at [0,0] (base)
        self.joint1_pos = np.array([self.link1 * np.cos(self.joint_angles[0]), 
                                    self.link1 * np.sin(self.joint_angles[0])], DTYPE)

        self.joint2_pos = self.joint1_pos + np.array([self.link2 * np.cos(self.joint_angles[1]), 
                                                      self.link2 * np.sin(self.joint_angles[1])], DTYPE)

        self.gripper_pos = self.joint2_pos.copy()

        # Update dart while being held
        if self.gripping and dart is not None: 
            dart.pos = self.gripper_pos.copy()
            dart.angle = self.joint_angles[1] + self.gripper_relative_angle - np.pi/2


    def release(self, dart: Dart):
        # Determine CoM position vector (using absolute angles at each joint) 
        x = self.link1*np.cos(self.joint_angles[0]) + self.link2*np.cos(self.joint_angles[1])
        y = self.link1*np.sin(self.joint_angles[0]) + self.link2*np.sin(self.joint_angles[1])
        dart.pos = np.array([x, y], DTYPE)

        # Determine CoM velocity vector
        xdot1 = -self.joint_angular_vels[0]*self.link1*np.cos(self.joint_angles[0] - np.pi/2)
        xdot2 = xdot1 - self.joint_angular_vels[1]*self.link2*np.cos(self.joint_angles[1] - np.pi/2)
        ydot1 = -self.joint_angular_vels[0]*self.link1*np.sin(self.joint_angles[0] - np.pi/2) 
        ydot2 = ydot1 - self.joint_angular_vels[1]*self.link2*np.sin(self.joint_angles[1] - np.pi/2) 
        dart.vel = np.array([xdot2, ydot2], DTYPE)
        print("relase:", dart.vel)

        # Determine angle of attack (using absolute angles at each joint and relative at gripper)
        dart.angle = self.joint_angles[1] + self.gripper_relative_angle - np.pi/2

        dart.state = np.concatenate([dart.pos, dart.vel, np.array([dart.angle], DTYPE), np.array([0.0], DTYPE)], dtype=DTYPE)


def erk4_step(func, t, state, dt, params):
    """A single step of the ERK4 method for ODEs."""
    k1 = func(t, state, params)
    k2 = func(t + 0.5*dt, state + 0.5*dt*k1, params)
    k3 = func(t + 0.5*dt, state + 0.5*dt*k2, params)
    k4 = func(t + dt, state + dt*k3, params)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def draw_arm_and_dart(robot_arm: RobotArm, dart: Dart):
    plt.figure(figsize=(10,10))
    plt.plot([robot_arm.joint0_pos[0], robot_arm.joint1_pos[0], robot_arm.joint2_pos[0]], 
             [robot_arm.joint0_pos[1], robot_arm.joint1_pos[1], robot_arm.joint2_pos[1]], 
             '-o', label='Robot Arm')
    plt.scatter(dart.pos[0], dart.pos[1], color='red', marker='x', label='Dart Position')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Arm and Dart')
    plt.show()


def animate_robot_arm_dart(robot_arm: RobotArm, dart: Dart, dt, T, angular_velocities, release_frame=13):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)

    robot_line, = ax.plot([], [], '-o', label='Robot Arm')
    
    dart_arrow = FancyArrowPatch((0,0), (0, dart.length), mutation_scale=10, color='red')
    ax.add_patch(dart_arrow)
    
    dart_released = False

    def init():
        robot_line.set_data([], [])
        dart_arrow.set_positions((0, 0), (0, 0))
        return robot_line, dart_arrow

    def update(frame):
        nonlocal dart_released
        if frame < release_frame:
            robot_arm.move_arm(dt, dart, angular_velocities[frame])
            if frame == release_frame - 1:
                robot_arm.release(dart)
        elif not dart_released:
            robot_arm.release(dart)
            dart_released = True
        else:
            dart.state = erk4_step(Dart.dart_flight_2d, frame*dt, dart.state, dt, dart.params)
            dart.pos = dart.state[:2]
            dart.angle = dart.state[4]

        # Set dart arrow orientation and position
        dx = np.cos(dart.angle)
        dy = np.sin(dart.angle)
        dart_arrow.set_positions((dart.pos[0], dart.pos[1]), (dart.pos[0] + dx, dart.pos[1] + dy))

        robot_line.set_data([robot_arm.joint0_pos[0], robot_arm.joint1_pos[0], robot_arm.joint2_pos[0]], 
                            [robot_arm.joint0_pos[1], robot_arm.joint1_pos[1], robot_arm.joint2_pos[1]])

        return robot_line, dart_arrow

    ani = animation.FuncAnimation(fig, update, frames=int(T/dt), init_func=init, blit=True)
    plt.legend()
    plt.grid(True)
    plt.show()


def print_member_values(obj):
    for attr, value in vars(obj).items():
        print(f"{attr}: {value}")

if __name__ == "__main__":
    robot_arm = RobotArm()
    dart = Dart()

    dt = 0.01
    T = 10 # Total simulation time
    angular_velocities = [-np.array([10, 10])] * int(T/dt) # Constant angular velocities for simulation

    animate_robot_arm_dart(robot_arm, dart, dt, T, angular_velocities)