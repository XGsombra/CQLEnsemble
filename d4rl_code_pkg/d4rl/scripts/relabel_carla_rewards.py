import h5py
import carla
import sys
import numpy as np

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle


class CustomGlobalRoutePlanner(GlobalRoutePlanner):
    def __init__(self, dao):
        super(CustomGlobalRoutePlanner, self).__init__(dao=dao)

    def compute_direction_velocities(self, origin, velocity, destination):
        origin = carla.Location(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)

        origin_xy = np.array([origin.x, origin.y])
        velocity_xy = np.array([velocity[0], velocity[1]])

        first_node_xy = self._graph.nodes[node_list[1]]['vertex']
        first_node_xy = np.array([first_node_xy[0], first_node_xy[1]])
        target_direction_vector = first_node_xy - origin_xy
        target_unit_vector = np.array(target_direction_vector) / np.linalg.norm(target_direction_vector)

        vel_s = np.dot(velocity_xy, target_unit_vector)

        unit_velocity = velocity_xy / (np.linalg.norm(velocity_xy) + 1e-8)
        angle = np.arccos(np.clip(np.dot(unit_velocity, target_unit_vector), -1.0, 1.0))
        vel_perp = np.linalg.norm(velocity_xy) * np.sin(angle)
        return vel_s, vel_perp

    def compute_distance(self, origin, destination):
        origin = carla.Location(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        node_list = super(CustomGlobalRoutePlanner, self)._path_search(origin=origin, destination=destination)
        #print('Node list:', node_list)
        first_node_xy = self._graph.nodes[node_list[0]]['vertex']
        #print('Diff:', origin, first_node_xy)

        #distance = 0.0
        distances = []
        distances.append(np.linalg.norm(np.array([origin.x, origin.y, 0.0]) - np.array(first_node_xy)))

        for idx in range(len(node_list) - 1):
            distances.append(super(CustomGlobalRoutePlanner, self)._distance_heuristic(node_list[idx], node_list[idx+1]))
        #print('Distances:', distances)
        #import pdb; pdb.set_trace()
        return np.sum(distances)

def goal_reaching_reward(vehicle_location, vehicle_velocity, target_location, collision_reward, route_planner):
    # This is the distance computation
    dist = route_planner.compute_distance(vehicle_location, target_location)
    vel_forward, vel_perp = route_planner.compute_direction_velocities(vehicle_location, vehicle_velocity, target_location)
    
    #print('[GoalReachReward] VehLoc: %s Target: %s Dist: %s VelF:%s' % (str(vehicle_location), str(target_location), str(dist), str(vel_forward)))

    #base_reward = -1.0 * (dist / 100.0) + 5.0
    base_reward = vel_forward
    #collided_done, collision_reward = self._get_collision_reward(vehicle)
    #traffic_light_done, traffic_light_reward = self._get_traffic_light_reward(vehicle)
    #object_collided_done, object_collided_reward = self._get_object_collided_reward(vehicle)
    total_reward = base_reward + 100 * collision_reward # + 100 * traffic_light_reward + 100.0 * object_collided_reward
    return total_reward

def setup():
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    _map = world.get_map()
    route_planner_dao = GlobalRoutePlannerDAO(_map, sampling_resolution=0.1) 
    route_planner = CustomGlobalRoutePlanner(route_planner_dao)
    route_planner.setup()
    target_location = carla.Location(x=-13.473097, y=134.311234, z=-0.010433)


    FILES = ['output_%d' % i for i in range(0,12)]

    for hfile in FILES:
        print('relabeling', hfile)
        h5data = h5py.File(hfile+'.hdf5', 'r+')
        #output_file = h5py.File(hfile+'_relabel.hdf5', 'w')
        N = h5data['rewards'].shape[0]
        new_rewards = []
        for i in range(N):
            loc = h5data['infos/location'][i]
            vel = h5data['infos/velocity'][i]
            collision_reward = h5data['infos/reward_collision'][i]
            new_reward = goal_reaching_reward(loc, vel, target_location, collision_reward, route_planner)
            new_rewards.append(new_reward)
            if i % 1001 == 0:
                print(i, new_reward)

        new_rewards = np.array(new_rewards)
        h5data['rewards'][...] = new_rewards

setup()
