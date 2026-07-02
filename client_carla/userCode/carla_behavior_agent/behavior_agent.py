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
        self._avoid_counter = 0

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

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

        vehicle_list = list(self._world.get_actors().filter("*vehicle*")) + \
               list(self._world.get_actors().filter("*static.prop*"))
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

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

    def _obstacle_avoid_manager(self, waypoint, vehicle, distance):
            """
            Builds an explicit bypass plan around a static blocking obstacle
            by stepping waypoints into the adjacent lane and back.

                :param waypoint:  current ego carla.Waypoint
                :param vehicle:   the blocking actor
                :param distance:  net distance to the obstacle front face (meters)
                :return: carla.VehicleControl from the new plan, or None if no bypass found
            """
            if self._speed > 5.0:
                return None
            if distance > 20.0:
                return None

            # How long the obstacle footprint is (use bounding box)
            obstacle_len = max(
                vehicle.bounding_box.extent.x,
                vehicle.bounding_box.extent.y
            ) * 2.0  # full length, not half

            # Distances along the current lane for each phase
            step = 2.0                               # waypoint resolution (meters)
            d_approach  = max(distance - 2.0, 1.0)  # stay just behind the obstacle
            d_bypass    = d_approach + obstacle_len + 6.0   # length of the bypass lane segment
            d_return    = d_bypass   + 6.0                  # merge back after clearing

            all_vehicles = list(self._world.get_actors().filter("*vehicle*"))

            for lane_offset, side in [(-1, "left"), (1, "right")]:
                # Sample a waypoint in the candidate bypass lane
                probe_wps = waypoint.next(d_approach + obstacle_len / 2)
                if not probe_wps:
                    continue
                probe_wp = probe_wps[0]

                if lane_offset == -1:
                    bypass_lane_wp = probe_wp.get_left_lane()
                else:
                    bypass_lane_wp = probe_wp.get_right_lane()

                if bypass_lane_wp is None:
                    continue
                if bypass_lane_wp.lane_type != carla.LaneType.Driving:
                    continue

                # Check bypass lane is clear of moving vehicles
                lane_blocked, _, _ = self._vehicle_obstacle_detected(
                    all_vehicles,
                    max_distance=d_return + 10.0,
                    up_angle_th=180,
                    lane_offset=lane_offset
                )
                if lane_blocked:
                    continue

                # --- Build the plan explicitly as a waypoint list ---
                plan = []

                # Phase 1: approach in current lane up to just before obstacle
                wps_approach = waypoint.next_until_lane_end(step)
                current_wp = waypoint
                dist_covered = 0.0
                while dist_covered < d_approach:
                    nexts = current_wp.next(step)
                    if not nexts:
                        break
                    current_wp = nexts[0]
                    dist_covered += step
                    plan.append((current_wp, RoadOption.LANEFOLLOW))

                # Phase 2: lateral move into bypass lane
                if lane_offset == -1:
                    side_wp = current_wp.get_left_lane()
                    lane_road_option = RoadOption.CHANGELANELEFT
                else:
                    side_wp = current_wp.get_right_lane()
                    lane_road_option = RoadOption.CHANGELANERIGHT

                if side_wp is None or side_wp.lane_type != carla.LaneType.Driving:
                    continue

                plan.append((side_wp, lane_road_option))

                # Phase 3: drive through the bypass lane past the obstacle
                current_wp = side_wp
                dist_covered = 0.0
                while dist_covered < (obstacle_len + 8.0):
                    nexts = current_wp.next(step)
                    if not nexts:
                        break
                    current_wp = nexts[0]
                    dist_covered += step
                    plan.append((current_wp, RoadOption.LANEFOLLOW))

                # Phase 4: merge back into original lane
                if lane_offset == -1:
                    return_wp = current_wp.get_right_lane()
                    return_option = RoadOption.CHANGELANERIGHT
                else:
                    return_wp = current_wp.get_left_lane()
                    return_option = RoadOption.CHANGELANELEFT

                if return_wp is None or return_wp.lane_type != carla.LaneType.Driving:
                    continue

                plan.append((return_wp, return_option))

                # Phase 5: continue a few meters in original lane to stabilize
                current_wp = return_wp
                for _ in range(int(10.0 / step)):
                    nexts = current_wp.next(step)
                    if not nexts:
                        break
                    current_wp = nexts[0]
                    plan.append((current_wp, RoadOption.LANEFOLLOW))

                if not plan:
                    continue

                print(f"[AvoidManager] Bypassing '{vehicle.type_id}' on the {side} "
                    f"| plan={len(plan)} wps | obstacle_len={obstacle_len:.1f}m")

                # Inject the bypass plan — clean_queue=True replaces whatever was queued
                self._local_planner.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
                return self._local_planner.run_step()

            print("[AvoidManager] No valid bypass found — emergency stop")
            return None

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
        if self._avoid_counter > 0:               
            self._avoid_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            
            print(f"[DEBUG] vehicle={vehicle.type_id if vehicle else None} distance={distance:.1f} "
          f"braking_distance={self._behavior.braking_distance} avoid_counter={self._avoid_counter}")

            if distance < self._behavior.braking_distance:
                if 'static.prop' in vehicle.type_id and self._avoid_counter == 0:
                    bypass_control = self._obstacle_avoid_manager(ego_vehicle_wp, vehicle, distance)
                    if bypass_control is not None:
                        self._avoid_counter = 150 
                        return bypass_control
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([self._behavior.max_speed, self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist])
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
