#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from behavior_agent import BehaviorAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import importlib

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

import json
from utils import Streamer

def get_entry_point():
    return 'MyTeamAgent'

class MyTeamAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self._agent = None
        
        with open(path_to_conf_file, "r") as f:
            self.configs = json.load(f)
            f.close()
        
        self.__show = len(self.configs["Visualizer_IP"]) > 0
        
        if self.__show:
            self.showServer = Streamer(self.configs["Visualizer_IP"])

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """
        return self.configs["sensors"]
    
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self._global_plan_world_coord = global_plan_world_coord
        self._global_plan = global_plan_gps

    def run_step(self, input_data, timestamp):
        if not self._agent:
            # Find the ego (hero) vehicle
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and \
                actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
            if not hero_actor:
                return carla.VehicleControl()

            # Initialize the BehaviorAgent
            self._agent = BehaviorAgent(hero_actor,
                                        behavior='normal',
                                        opt_dict=self.configs)

            # Convert the global plan to waypoints and set it
            plan = []
            prev_wp = None
            for transform, _ in self._global_plan_world_coord:
                wp = CarlaDataProvider.get_map() \
                        .get_waypoint(transform.location)
                if prev_wp:
                    plan.extend(self._agent.trace_route(prev_wp, wp))
                prev_wp = wp
            self._agent.set_global_plan(plan)
            return carla.VehicleControl()

        return self._agent.run_step()

    def destroy(self):
        print("DESTROY")
        if self._agent:
            self._agent.reset()
            