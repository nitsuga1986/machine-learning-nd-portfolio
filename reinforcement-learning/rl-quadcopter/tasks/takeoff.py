import numpy as np
from physics_sim import PhysicsSim
from scipy.spatial import distance

class Task_takeoff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        euc_distance = distance.euclidean(self.sim.pose[:3] , self.target_pos)
        euc_distance_z = distance.euclidean(self.sim.pose[2] , self.target_pos[2])
        angular_deviation = np.sqrt(np.square(self.sim.pose[3:]).sum())
        # Constant reward
        reward = 1
        # Reward for remain close target
        if euc_distance < 2:
            reward += 2
        # Penalty
        penalty = euc_distance**2 + euc_distance_z**2 + 2* angular_deviation
        # Final reward
        reward = reward - 0.002 * penalty
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            # penalize crash
            if done and self.sim.time < self.sim.runtime:
                reward = -1
            pose_all.append(self.sim.pose)
            pose_all.append(self.sim.angular_v)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        pose_all = np.append(self.sim.pose, self.sim.angular_v)
        state = np.concatenate([pose_all] * self.action_repeat) 
        return state