# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal

from misc import get_speed, positive, is_within_distance, compute_distance

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    OBSTACLE_MAX_DISTANCE = 45
    FORWARD_ANGLE_STRAIGHT = 30
    FORWARD_ANGLE_TURN = 60

    BYPASS_DETECTION_DISTANCE = 80
    BYPASS_MIN_GAP_TIME = 4.0

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        # Cycle : 'idle' -> 'waiting_gap' -> 'overtaking' -> 'returning' -> 'idle'
        self._bypass_state = 'idle'
        self._bypass_origin_waypoint = None

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

#----------------------------------------------------------------------------------------------#

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def _build_obstacle_list(self, waypoint, max_distance=None):
        """
        Builds the extended list of obstacles to monitor: vehicles
        (including cyclists, who are ‘vehicle’-type agents in
        CARLA) and static objects (e.g. a DynamicObjectCrossing container).

            :param waypoint: the agent’s current waypoint
            :param max_distance: detection range (default OBSTACLE_MAX_DISTANCE)
            :return: list of obstacle agents within the given range
        """
        max_distance = self.OBSTACLE_MAX_DISTANCE if max_distance is None else max_distance
        actors = self._world.get_actors()
        obstacle_list = list(actors.filter("*vehicle*")) + list(actors.filter("*static.prop*"))

        def dist(v): return v.get_location().distance(waypoint.transform.location)
        return [v for v in obstacle_list if dist(v) < max_distance and v.id != self._vehicle.id]

    def _is_in_turn(self):
        """
        Indicates whether the vehicle is approaching or negotiating a bend.

            :return: True if the direction of travel is a left/right bend
        """
        return self._incoming_direction in (RoadOption.LEFT, RoadOption.RIGHT)

    def _forward_detection_angle(self):
        """
        Frontal detection angle to be used: widened when cornering to detect
        lateral obstacles (cyclists, encroaching vehicles,
        crossing objects) earlier.

            :return: detection angle (degrees)
        """
        return self.FORWARD_ANGLE_TURN if self._is_in_turn() else self.FORWARD_ANGLE_STRAIGHT

    def _lane_change_obstacle_detected(self, vehicle_list, lane_offset):
        """
        Detects an obstacle occupying the target lane during a lane change.

            :param vehicle_list: list of obstacles to consider
            :param lane_offset: -1 for the left-hand lane, 1 for the right-hand lane
            :return: tuple (vehicle_state, vehicle, distance)
        """
        return self._vehicle_obstacle_detected(
            vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2),
            up_angle_th=180, lane_offset=lane_offset)

    def _forward_obstacle_detected(self, vehicle_list):
        """
        Detects an obstacle approaching head-on, with an expanded detection angle
        when cornering (see `_forward_detection_angle`).

            :param vehicle_list: list of obstacles to consider
            :return: tuple (vehicle_state, vehicle, distance)
        """
        return self._vehicle_obstacle_detected(
            vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3),
            up_angle_th=self._forward_detection_angle())

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._build_obstacle_list(waypoint)

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._lane_change_obstacle_detected(vehicle_list, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._lane_change_obstacle_detected(vehicle_list, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._forward_obstacle_detected(vehicle_list)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

#----------------------------------------------------------------------------------------------#

    def _static_obstacle_ahead(self, waypoint):
        """
        Detects a static obstacle (roadworks, accident, parked vehicle)
        blocking the agent’s path, within the detour range.

            :param waypoint: the agent’s current waypoint
            :return: tuple (obstacle_state, obstacle, distance)
        """
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        static_list = [o for o in obstacle_list if "static.prop" in o.type_id]
        return self._vehicle_obstacle_detected(
            static_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=self.FORWARD_ANGLE_STRAIGHT)

    def _oncoming_lane_obstacle(self, waypoint):
        """
        Detects a vehicle travelling in the opposite direction on the opposite carriageway,
        used to search for a gap before overtaking.

            :param waypoint: the agent’s current waypoint
            :return: tuple (vehicle_state, vehicle, distance)
        """
        actors = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in actors if dist(v) < self.BYPASS_DETECTION_DISTANCE and v.id != self._vehicle.id]
        return self._vehicle_obstacle_detected(
            vehicle_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=180, lane_offset=-1)

    def _gap_is_safe(self, oncoming_distance, oncoming_speed):
        """
        Assesses whether there is a sufficient gap in oncoming traffic to
        merge (gap detection / gap acceptance).

            :param oncoming_distance: distance to the oncoming vehicle (m), None/-1 if not available
            :param oncoming_speed: speed of the oncoming vehicle (km/h)
            :return: True if the gap is deemed safe
        """
        if oncoming_distance is None or oncoming_distance < 0:
            return True  # no vehiclle detected : free line

        speed_ms = oncoming_speed / 3.6
        if speed_ms <= 0:
            return True  # vehicle stopped : no imminent risk of frontal collision

        time_to_arrival = oncoming_distance / speed_ms
        return time_to_arrival >= self.BYPASS_MIN_GAP_TIME

    def _can_start_bypass(self, waypoint):
        """
        Determines whether the agent can move onto the opposite lane to
        go round the detected obstacle (if the lane is clear or there is sufficient space).

            :param waypoint: the agent’s current waypoint
            :return: True if the manoeuvre can begin
        """
        oncoming_state, oncoming_vehicle, oncoming_distance = self._oncoming_lane_obstacle(waypoint)
        if not oncoming_state:
            return True
        return self._gap_is_safe(oncoming_distance, get_speed(oncoming_vehicle))

    def _start_bypass_maneuver(self, waypoint):
        """
        Triggers a lateral shift to the opposite lane to bypass the obstacle 
        (same mechanism as `_tailgating`: redefine the local destination to the adjacent lane).

            :param waypoint: the agent’s current waypoint
        """
        opposite_wpt = waypoint.get_left_lane()
        end_waypoint = self._local_planner.target_waypoint
        self.set_destination(end_waypoint.transform.location, opposite_wpt.transform.location)

    def _obstacle_cleared(self, waypoint):
        """
        Indicates whether the obstacle that was bypassed has now been passed 
        (no longer detected in front of the agent).

            :param waypoint: the agent’s current waypoint
            :return: True if the obstacle is no longer a frontal obstacle
        """
        obstacle_state, _, _ = self._static_obstacle_ahead(waypoint)
        return not obstacle_state

    def _back_on_original_lane(self, waypoint):
        """
        Indicates whether the agent has returned to its original path after taking a detour.

            :param waypoint: the agent’s current waypoint
            :return: True if the current path matches the starting path
        """
        return (self._bypass_origin_waypoint is not None
                and waypoint.lane_id == self._bypass_origin_waypoint.lane_id)

    def _reset_bypass_state(self):
        """Resets the status of the bypass module."""
        self._bypass_state = 'idle'
        self._bypass_origin_waypoint = None

    def bypass_obstacle_manager(self, waypoint):
        """
        Generic module for static obstacle avoidance. Drives the cycle of
        detection → waiting for a gap → avoidance → return
        to the route.

            :param waypoint: the agent’s current waypoint
            :return: True if a bypass manoeuvre is in progress (the caller
                     must then allow the local planner to follow the new
                     destination rather than applying the normal behaviour)
        """
        if self._bypass_state == 'idle':
            obstacle_state, _, _ = self._static_obstacle_ahead(waypoint)
            if obstacle_state:
                self._bypass_origin_waypoint = waypoint
                self._bypass_state = 'waiting_gap'
            return self._bypass_state != 'idle'

        if self._bypass_state == 'waiting_gap':
            if self._can_start_bypass(waypoint):
                self._start_bypass_maneuver(waypoint)
                self._bypass_state = 'overtaking'
            return True

        if self._bypass_state == 'overtaking':
            if self._obstacle_cleared(waypoint):
                self._bypass_state = 'returning'
            return True

        if self._bypass_state == 'returning':
            if self._back_on_original_lane(waypoint):
                self._reset_bypass_state()
                return False
            return True

        return False

#----------------------------------------------------------------------------------------------#

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

#----------------------------------------------------------------------------------------------#

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

#----------------------------------------------------------------------------------------------#

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.2: Static obstacle bypass behavior (construction/accident/parked vehicle)
        if self.bypass_obstacle_manager(ego_vehicle_wp):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            return self._local_planner.run_step(debug=debug)

        # 2.3: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control