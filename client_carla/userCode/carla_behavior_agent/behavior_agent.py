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
        self._overtake_state = 'IDLE'

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
    #     if self._speed > 5.0:
    #         return None
    #     if distance > 20.0:
    #         return None

    #     step = 2.0
    #     d_approach = max(distance - 8.0, 1.0)
    #     d_through  = 20.0
    #     return_stabilize = 10.0

    #     all_vehicles = list(self._world.get_actors().filter("*vehicle*"))

    #     for lane_offset, side in [(-1, "left"), (1, "right")]:
    #         probe_wps = waypoint.next(d_approach + d_through / 2)
    #         if not probe_wps:
    #             print(f"[AvoidManager] {side}: no waypoint ahead")
    #             continue
    #         probe_wp = probe_wps[0]

    #         bypass_lane_wp = probe_wp.get_left_lane() if lane_offset == -1 else probe_wp.get_right_lane()
    #         if bypass_lane_wp is None:
    #             print(f"[AvoidManager] {side}: no adjacent lane exists")
    #             continue
    #         if bypass_lane_wp.lane_type != carla.LaneType.Driving:
    #             print(f"[AvoidManager] {side}: adjacent lane not drivable (type={bypass_lane_wp.lane_type})")
    #             continue

    #         # Detecte si la voie cible va dans le sens opposé au nôtre
    #         opposite_direction = (waypoint.lane_id * bypass_lane_wp.lane_id) < 0
    #         print(f"[AvoidManager] {side}: opposite_direction={opposite_direction} "
    #             f"(ego lane_id={waypoint.lane_id}, target lane_id={bypass_lane_wp.lane_id})")

    #         lane_blocked, blocker, blocker_dist = self._vehicle_obstacle_detected(
    #             all_vehicles,
    #             max_distance=d_approach + d_through + 10.0,
    #             up_angle_th=180,
    #             lane_offset=lane_offset
    #         )
    #         if lane_blocked:
    #             bname = blocker.type_id if blocker else "?"
    #             print(f"[AvoidManager] {side}: blocked by {bname}")
    #             continue

    #         plan = []

    #         # Phase 1: approche dans la voie d'origine (toujours .next(), c'est notre propre sens)
    #         orig_wp = waypoint
    #         dist_covered = 0.0
    #         while dist_covered < d_approach:
    #             nexts = orig_wp.next(step)
    #             if not nexts:
    #                 break
    #             orig_wp = nexts[0]
    #             dist_covered += step
    #             plan.append((orig_wp, RoadOption.LANEFOLLOW))

    #         # Phase 2: bascule dans la voie opposée
    #         side_wp = orig_wp.get_left_lane() if lane_offset == -1 else orig_wp.get_right_lane()
    #         lane_road_option = RoadOption.CHANGELANELEFT if lane_offset == -1 else RoadOption.CHANGELANERIGHT
    #         if side_wp is None or side_wp.lane_type != carla.LaneType.Driving:
    #             print(f"[AvoidManager] {side}: lateral move failed")
    #             continue
    #         plan.append((side_wp, lane_road_option))

    #         # Phase 3: on avance physiquement vers l'avant, ce qui veut dire
    #         # .previous() si la voie cible est de sens opposé, .next() sinon
    #         current_wp = side_wp
    #         dist_covered = 0.0
    #         while dist_covered < d_through:
    #             if opposite_direction:
    #                 nexts = current_wp.previous(step)
    #             else:
    #                 nexts = current_wp.next(step)
    #             if not nexts:
    #                 break
    #             current_wp = nexts[0]
    #             dist_covered += step
    #             plan.append((current_wp, RoadOption.LANEFOLLOW))

    #             orig_nexts = orig_wp.next(step)
    #             if orig_nexts:
    #                 orig_wp = orig_nexts[0]

    #         # Phase 4: retour dans la voie d'origine
    #         if orig_wp is None or orig_wp.lane_type != carla.LaneType.Driving:
    #             print(f"[AvoidManager] {side}: merge-back failed (tracked original lane invalid)")
    #             continue
    #         return_option = RoadOption.CHANGELANERIGHT if lane_offset == -1 else RoadOption.CHANGELANELEFT
    #         plan.append((orig_wp, return_option))

    #         # Phase 5: stabilisation
    #         current_wp = orig_wp
    #         for _ in range(int(return_stabilize / step)):
    #             nexts = current_wp.next(step)
    #             if not nexts:
    #                 break
    #             current_wp = nexts[0]
    #             plan.append((current_wp, RoadOption.LANEFOLLOW))

    #         if not plan:
    #             continue

    #         print(f"[AvoidManager] Bypassing on the {side} | plan={len(plan)} wps | d_through={d_through:.1f}m")

    #         self._local_planner.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
    #         return self._local_planner.run_step()

    #     print("[AvoidManager] No valid bypass found — emergency stop")
         return None

    def _obstacle_still_present(self, road_id, lane_id, max_check_distance=25.0):
        all_props = self._world.get_actors().filter("static.prop.*")
        ego_loc = self._vehicle.get_location()
        for prop in all_props:
            prop_wp = self._map.get_waypoint(prop.get_location(), lane_type=carla.LaneType.Any)
            if prop_wp.road_id == road_id and prop_wp.lane_id == lane_id:
                if prop.get_location().distance(ego_loc) < max_check_distance:
                    return True
        return False

    def _oncoming_lane_clear(self, lane_offset=-1, max_distance=40.0):
        """
        Vérifie en temps réel si la voie opposée est libre de tout véhicule venant
        en face. Doit être appelée à CHAQUE tick pendant tout le dépassement,
        pas seulement au moment de décider de le commencer.
        """
        all_vehicles = list(self._world.get_actors().filter("*vehicle*"))
        lane_blocked, blocker, blocker_dist = self._vehicle_obstacle_detected(
            all_vehicles, max_distance=max_distance, up_angle_th=180, lane_offset=lane_offset
        )
        return not lane_blocked, blocker

    def overtake_manager(self, ego_wp, vehicle=None, distance=None):
        """
        Contrôleur réactif de dépassement. Appelé à chaque tick tant qu'un
        dépassement est en cours ou nécessaire. Contrairement à l'ancien plan
        figé, il réévalue à chaque tick :
        - si un véhicule arrive en face (abandon/pause du dépassement)
        - si l'obstacle d'origine est encore là (moment de rentrer dans la voie)

        États : 'IDLE' -> 'CROSSING' -> 'MERGING_BACK' -> 'IDLE'
        """

        # --- État 1 : pas encore en dépassement, on évalue s'il faut en démarrer un ---
        if self._overtake_state == 'IDLE':
            if vehicle is None or 'static.prop' not in vehicle.type_id:
                return None
            if self._speed > 5.0 or distance > 20.0:
                return None

            clear, blocker = self._oncoming_lane_clear(lane_offset=-1)
            if not clear:
                print(f"[Overtake] En attente : véhicule en face ({blocker.type_id if blocker else '?'})")
                return self.emergency_stop()

            target_wp = ego_wp.get_left_lane()
            if target_wp is None or target_wp.lane_type != carla.LaneType.Driving:
                print("[Overtake] Pas de voie opposée exploitable")
                return None

            self._overtake_state = 'CROSSING'
            self._original_lane_id = ego_wp.lane_id 
            self._original_road_id = ego_wp.road_id
            print("[Overtake] Départ du dépassement sur la gauche")
            # continue directement en CROSSING sur ce même tick

        # --- État 2 : on roule sur la voie opposée, tant que l'obstacle est encore là ---
        if self._overtake_state == 'CROSSING':
            clear, blocker = self._oncoming_lane_clear(lane_offset=-1, max_distance=25.0)
            if not clear:
                print(f"[Overtake] Véhicule en face détecté en pleine traversée "
                    f"({blocker.type_id if blocker else '?'}) — freinage")
                return self.emergency_stop()

            current_ego_wp = self._map.get_waypoint(self._vehicle.get_location())
            left_wp = current_ego_wp.get_left_lane()
            if left_wp is None or left_wp.lane_type != carla.LaneType.Driving:
                left_wp = current_ego_wp

            next_wps = left_wp.next(4.0)
            target = next_wps[0] if next_wps else left_wp
            self._local_planner.set_global_plan(
                [(target, RoadOption.CHANGELANELEFT)], stop_waypoint_creation=True, clean_queue=True)

            still_present = self._obstacle_still_present(self._original_road_id, self._original_lane_id)
            
            control = self._local_planner.run_step()
            self._crossing_tick_counter = getattr(self, '_crossing_tick_counter', 0) + 1
            if self._crossing_tick_counter % 20 == 0:
                print(f"[Overtake][CROSSING] speed={self._speed:.1f} "
                    f"steer={control.steer:.3f} throttle={control.throttle:.2f} "
                    f"still_present={still_present}")

            if not still_present:
                self._overtake_state = 'MERGING_BACK'
                self._crossing_tick_counter = 0
                print("[Overtake] Obstacle dépassé — retour sur la voie d'origine")

            return control

        # --- État 3 : on revient sur la voie d'origine ---
        if self._overtake_state == 'MERGING_BACK':
            current_ego_wp = self._map.get_waypoint(self._vehicle.get_location())
            right_wp = current_ego_wp.get_right_lane()

            if right_wp is not None and right_wp.lane_type == carla.LaneType.Driving \
                    and current_ego_wp.lane_id == right_wp.lane_id:
                self._overtake_state = 'IDLE'
                print("[Overtake] Retour terminé")
                return None

            if right_wp is None or right_wp.lane_type != carla.LaneType.Driving:
                right_wp = current_ego_wp

            next_wps = right_wp.next(4.0)
            target = next_wps[0] if next_wps else right_wp
            self._local_planner.set_global_plan(
                [(target, RoadOption.CHANGELANERIGHT)], stop_waypoint_creation=True, clean_queue=True)
            return self._local_planner.run_step()

        return None

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()
        loc = self._vehicle.get_location()
        vel = self._vehicle.get_velocity()
        self._tick_counter_global = getattr(self, '_tick_counter_global', 0) + 1
        if self._tick_counter_global % 10 == 0:
            print(f"[Position] tick={self._tick_counter_global} "
                f"x={loc.x:.2f} y={loc.y:.2f} z={loc.z:.2f} "
                f"vel=({vel.x:.2f},{vel.y:.2f},{vel.z:.2f}) speed_reported={self._speed:.2f}")

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1
        if self._overtake_state != 'IDLE':
            ego_vehicle_wp = self._map.get_waypoint(self._vehicle.get_location())
            result = self.overtake_manager(ego_vehicle_wp)
            if result is not None:
                return result

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

            result = self.overtake_manager(ego_vehicle_wp, vehicle, distance)
            if result is not None:
                return result

            if distance < self._behavior.braking_distance:
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
