# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import math
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

    DETECTION_SPEED_MARGIN_SECONDS = 2.5 
    BRAKING_SPEED_MARGIN_SECONDS = 0.8

    LATERAL_HAZARD_HALF_WIDTH = 3.0

    CYCLIST_TYPE_KEYWORDS = (
        'vehicle.bh.crossbike',
        'vehicle.diamondback.century',
        'vehicle.gazelle.omafiets',
    )

    # Lateral clearance kept from a cyclist ahead (HazardAtSideLane), in
    # metres. Small enough to stay well within the lane (~3.5 m wide).
    CYCLIST_CLEARANCE_OFFSET = 0.5

    JUNCTION_STALL_EXCLUSION_DISTANCE = 25.0   

    BYPASS_DETECTION_DISTANCE = 80
    BYPASS_MIN_GAP_TIME = 4.0
    BYPASS_LANE_OVERLAP_MARGIN = 0.3

    BYPASS_ABORT_RESUME_SPEED_THRESHOLD = 8.0   
    BYPASS_ABORT_RESUME_TICKS = 40   

    BYPASS_NO_PROGRESS_TICKS = 100      

    STATE_LOG_INTERVAL = 20
    BYPASS_TIMEOUT_TICKS = 800

    JUNCTION_DETECTION_DISTANCE = 30
    JUNCTION_MIN_GAP_TIME = 3.0
    JUNCTION_TIMEOUT_TICKS = 300

    STOP_SIGN_SPEED_EPSILON = 0.5       

    STALL_SPEED_THRESHOLD = 1.0    
    STALL_TIMEOUT_TICKS = 150      
    EGO_BLOCKED_SPEED_THRESHOLD = 2.0    
    EGO_BLOCKED_CONFIRM_TICKS = 100 
    BYPASS_MAX_CONSECUTIVE_RETRIES = 2
    BYPASS_GIVEUP_COOLDOWN_TICKS = 300   

    STALL_MIN_EGO_MOVED_SPEED_KMH = 10.0   
    STALL_STARTUP_HARD_BLOCK_TICKS = 400   
    STALL_STARTUP_GRACE_TICKS = 200   

    STALL_VEHICLE_MOVED_SPEED_KMH = 5.0

    PEDESTRIAN_WAIT_TIMEOUT_TICKS = 200   
    PEDESTRIAN_STATIONARY_SPEED = 0.5     
    PEDESTRIAN_CREEP_SPEED = 5            

    CONTROL_LOSS_HEADING_THRESHOLD = 25.0  
    CONTROL_LOSS_RECOVERY_THRESHOLD = 8.0   
    CONTROL_LOSS_MIN_SPEED_KMH = 5.0        
    CONTROL_LOSS_STABILIZE_SPEED = 20.0     
    CONTROL_LOSS_TIMEOUT_TICKS = 400        
    CONTROL_LOSS_DEBOUNCE_TICKS = 3
    CONTROL_LOSS_BYPASS_GRACE_TICKS = 15

    WET_HEADING_MARGIN = 0.7
    WET_SPEED_MARGIN = 0.7

#----------------------------------------------------------------------------------------------#
#   INIT
#----------------------------------------------------------------------------------------------#

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

        # Minimal bypass flags (the full bypass rework is handled separately).
        # These must exist so run_step's bypass step doesn't raise before the
        # tick can reach the car-following / cyclist-clearance logic.
        self._bypassing = False
        self._bypassed_vehicle = None

        # Cycle : 'idle' -> 'waiting_gap' -> 'overtaking' -> 'returning' -> 'idle'
        self._bypass_state = 'idle'
        self._bypass_origin_waypoint = None
        self._bypass_start_tick = None
        self._last_bypass_transition_tick = None  
        self._bypass_target_actor = None       
        self._bypass_target_resumed_tick_counter = 0  
        self._bypass_best_offset = None   
        self._bypass_best_offset_tick = None

        self._actors = None

        self._tick_count = 0
        self._scenario_result = None

        # Cycle : 'idle' -> 'waiting_clear' -> 'crossing' -> 'idle'
        self._junction_state = 'idle'
        self._junction_start_tick = None

        # Stop-sign state 
        self._target_stop_sign = None
        self._stop_sign_done_id = None
        self._stop_map = {}

        # Stalled-vehicle tracking
        self._stalled_vehicle_id = None
        self._stalled_tick_counter = 0
        self._ego_blocked_tick_counter = 0
        self._ego_has_moved_normally = False  
        self._vehicles_seen_moving = set()     

        self._bypass_giveup_id = None
        self._bypass_giveup_tick = None
        self._last_bypass_failed_id = None
        self._bypass_retry_count = 0

        # Pedestrian wait tracking
        self._pedestrian_wait_id = None
        self._pedestrian_wait_tick_counter = 0

        # Cycle : 'idle' -> 'stabilizing' -> 'idle'
        self._control_loss_state = 'idle'
        self._control_loss_start_tick = None
        self._control_loss_pending_ticks = 0

#----------------------------------------------------------------------------------------------#
#   LOGS
#----------------------------------------------------------------------------------------------#

    def _refresh_actor_snapshot(self):
        """
        Retrieves all actors in the scene just once per tick.
        """
        self._actors = self._world.get_actors()

    def _track_moving_vehicles(self):

        if self._actors is None:
            return
        ego_location = self._vehicle.get_location()
        for vehicle in self._actors.filter("*vehicle*"):
            if vehicle.id == self._vehicle.id:
                continue
            if vehicle.get_location().distance(ego_location) > self.BYPASS_DETECTION_DISTANCE:
                continue
            if get_speed(vehicle) > self.STALL_VEHICLE_MOVED_SPEED_KMH:
                self._vehicles_seen_moving.add(vehicle.id)

    def _log_stop_sign_transition(self, event, stop_sign=None):

        sid = f" id={stop_sign.id}" if stop_sign is not None else ""
        messages = {
            'stopping': f"[STOP] Stop sign ahead{sid} -> braking to a full stop",
            'cleared':  f"[STOP] Full stop registered{sid} -> proceeding",
        }
        msg = messages.get(event)
        if msg is None:
            return
        if event == 'stopping' and (self._tick_count % self.STATE_LOG_INTERVAL) != 0:
            return
        print(msg)

    def _log_vehicle_state(self, waypoint):

        if self._tick_count % self.STATE_LOG_INTERVAL != 0:
            return
        loc = waypoint.transform.location
        print(f"[STATE] tick={self._tick_count} pos=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) "
              f"speed={self._speed:.1f} km/h")

    def _log_bypass_transition(self, event, waypoint=None, obstacle=None):

        loc = waypoint.transform.location if waypoint is not None else None
        pos = f" pos=({loc.x:.1f}, {loc.y:.1f})" if loc is not None else ""
        obstacle_info = f" obstacle={obstacle.type_id}(id={obstacle.id})" if obstacle is not None else ""
        messages = {
            'detected': f"[BYPASS] Obstacle detected{pos}{obstacle_info} -> searching for a gap",
            'gap_found': f"[BYPASS] Gap found{pos}{obstacle_info} -> start of manoeuvre",
            'success': f"[BYPASS] TEST SUCCESSFUL: obstacle bypassed, back on track{pos}",
            'timeout': f"[BYPASS] TEST FAILED: manoeuvre timed out{pos}"
                       f"(stuck in '{self._bypass_state}')",
            'no_lane': f"[BYPASS] TEST FAILED: no usable opposite lane{pos} -> abandoning bypass",
            'resumed_normal': f"[BYPASS] Target resumed normal driving{pos}{obstacle_info} "
                               f"-> was never really stalled, aborting manoeuvre and reverting "
                               f"to car-following",
            'cyclist_gone': f"[BYPASS] Tracked cyclist no longer detected{pos}{obstacle_info} "
                             f"-> aborting manoeuvre and reverting to normal behaviour",
            'no_progress': f"[BYPASS] TEST FAILED: overtaking made no progress{pos}{obstacle_info} "
                            f"-> obstacle never passed (deadlock/too slow), giving up and "
                            f"reverting to car-following",
        }
        print(messages[event])
        if event == 'success':
            self._scenario_result = True
        elif event in ('timeout', 'no_lane', 'no_progress'):
            self._scenario_result = False
        # 'resumed_normal' and 'cyclist_gone' are correct self-diagnoses, not scenario
        # failures - they deliberately leave self._scenario_result untouched.

    def _log_junction_transition(self, event, waypoint=None):

        loc = waypoint.transform.location if waypoint is not None else None
        pos = f" pos=({loc.x:.1f}, {loc.y:.1f})" if loc is not None else ""
        messages = {
            'waiting': f"[JUNCTION] Blocked or busy junction ahead{pos} -> waiting for a gap",
            'clear': f"[JUNCTION] Junction clear{pos} -> proceeding with the turn",
            'timeout': f"[JUNCTION] TEST FAILED: waited too long at the junction{pos} "
                       f"(stuck in '{self._junction_state}')",
        }
        print(messages[event])
        if event == 'timeout':
            self._scenario_result = False

    def _log_pedestrian_wait_timeout(self, waypoint=None):

        loc = waypoint.transform.location if waypoint is not None else None
        pos = f" pos=({loc.x:.1f}, {loc.y:.1f})" if loc is not None else ""
        print(f"[PEDESTRIAN] Waited too long for a stationary pedestrian{pos} -> creeping past cautiously")

    def _log_control_loss_transition(self, event, waypoint=None):

        loc = waypoint.transform.location if waypoint is not None else None
        pos = f" pos=({loc.x:.1f}, {loc.y:.1f})" if loc is not None else ""
        messages = {
            'detected': f"[CONTROL_LOSS] Uncommanded heading deviation{pos} -> stabilizing (throttle only)",
            'recovered': f"[CONTROL_LOSS] TEST SUCCESSFUL: heading back under control{pos}",
            'timeout': f"[CONTROL_LOSS] TEST FAILED: stabilization did not clear in time{pos}",
        }
        print(messages[event])
        if event == 'recovered':
            self._scenario_result = True
        elif event == 'timeout':
            self._scenario_result = False

    def _bypass_timed_out(self):

        return (self._tick_count - self._bypass_start_tick) > self.BYPASS_TIMEOUT_TICKS

    def _update_information(self):

        self._refresh_actor_snapshot()
        self._track_moving_vehicles()
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        # Default to no lateral offset each tick; the cyclist-clearance logic
        # re-applies one only when a cyclist is detected ahead.
        self._local_planner.set_offset(0)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

#----------------------------------------------------------------------------------------------#
#   TRAFFIC SIGNS
#----------------------------------------------------------------------------------------------#

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._actors
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def stop_sign_manager(self):
        """
        This method is in charge of behaviors for stop signs.
        """
        affected, stop_sign = self._affected_by_stop_sign()

        if not affected:
            # Left the trigger zone: re-arm for the next (or same) sign.
            self._target_stop_sign = None
            self._stop_sign_done_id = None
            return False

        # Already completed a full stop for this sign while still near it.
        if stop_sign.id == self._stop_sign_done_id:
            return False

        self._target_stop_sign = stop_sign.id

        if get_speed(self._vehicle) < self.STOP_SIGN_SPEED_EPSILON:
            self._stop_sign_done_id = stop_sign.id
            self._target_stop_sign = None
            self._log_stop_sign_transition('cleared', stop_sign)
            return False

        self._log_stop_sign_transition('stopping', stop_sign)
        return True

#----------------------------------------------------------------------------------------------#
#   COLLISION AND CAR AVOID
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
        
        max_distance = self.OBSTACLE_MAX_DISTANCE if max_distance is None else max_distance
        actors = self._actors
        obstacle_list = list(actors.filter("*vehicle*")) + list(actors.filter("*static.prop*"))

        def dist(v): return v.get_location().distance(waypoint.transform.location)
        return [v for v in obstacle_list if dist(v) < max_distance and v.id != self._vehicle.id]

    def _dynamic_forward_distance(self, base_distance):

        speed_ms = self._speed / 3.6
        return base_distance + speed_ms * self.DETECTION_SPEED_MARGIN_SECONDS

    def _effective_braking_distance(self):

        speed_ms = self._speed / 3.6
        return self._behavior.braking_distance + speed_ms * self.BRAKING_SPEED_MARGIN_SECONDS

    def _collision_detection_range(self):

        base = max(self.OBSTACLE_MAX_DISTANCE,
                   self._behavior.min_proximity_threshold,
                   self._speed_limit / 3)
        return self._dynamic_forward_distance(base)

    def _road_projection(self, actor, waypoint):

        yaw = math.radians(waypoint.transform.rotation.yaw)
        forward = (math.cos(yaw), math.sin(yaw))
        right = (-math.sin(yaw), math.cos(yaw))
        loc = actor.get_location()
        origin = waypoint.transform.location
        dx, dy = loc.x - origin.x, loc.y - origin.y
        longitudinal = dx * forward[0] + dy * forward[1]
        lateral = dx * right[0] + dy * right[1]
        return longitudinal, lateral

    def _lateral_hazard_ahead(self, waypoint):

        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self._collision_detection_range())
        detected, vehicle, distance = self._vehicle_obstacle_detected(
            obstacle_list, max_distance=30
        )
        if detected is False:
            return False, None, -1
        return True, vehicle, distance

    def _forward_detection_angle(self):

        return (
            self.FORWARD_ANGLE_TURN
            if self._incoming_direction in (RoadOption.LEFT, RoadOption.RIGHT)
            else self.FORWARD_ANGLE_STRAIGHT
        )

    def _lane_change_obstacle_detected(self, vehicle_list, lane_offset):

        return self._vehicle_obstacle_detected(
            vehicle_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2),
            up_angle_th=180, lane_offset=lane_offset)

    def _forward_obstacle_detected(self, vehicle_list):

        return self._vehicle_obstacle_detected(
            vehicle_list, self._collision_detection_range(),
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

        vehicle_list = self._build_obstacle_list(waypoint, max_distance=self._collision_detection_range())

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._lane_change_obstacle_detected(vehicle_list, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._lane_change_obstacle_detected(vehicle_list, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._forward_obstacle_detected(vehicle_list)

            lateral_state, lateral_actor, lateral_distance = self._lateral_hazard_ahead(waypoint)
            if lateral_state and (not vehicle_state or lateral_distance < distance):
                vehicle_state, vehicle, distance = lateral_state, lateral_actor, lateral_distance

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

#----------------------------------------------------------------------------------------------#

    def _bbox_adjusted_distance(self, actor, distance):
        """
        Converts a center-to-center distance into an edge-to-edge one using
        both actors' bounding boxes
        """
        return distance - max(
            actor.bounding_box.extent.y, actor.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

    def _react_to_moving_obstacle(self, waypoint, debug=False):

        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(waypoint)
        if not vehicle_state:
            return None
        if self._should_ignore_vehicle(vehicle):
            return None

        distance = self._bbox_adjusted_distance(vehicle, distance)
        if distance < self._effective_braking_distance():
            return self.emergency_stop()
        return self.car_following_manager(vehicle, distance, debug=debug)
    
    def _should_ignore_vehicle(self, vehicle):

        return (
            self._bypassing
            and self._bypassed_vehicle is not None
            and vehicle.id == self._bypassed_vehicle.id
        )

#----------------------------------------------------------------------------------------------#
# OBJECT DETECTION
#----------------------------------------------------------------------------------------------#

    def _is_obstacle_in_lane(self, obstacle, waypoint):

        _, lateral = self._road_projection(obstacle, waypoint)
        obstacle_half_width = max(obstacle.bounding_box.extent.x, obstacle.bounding_box.extent.y)
        return (abs(lateral) - obstacle_half_width) < (waypoint.lane_width / 2.0 + self.BYPASS_LANE_OVERLAP_MARGIN)


    def _is_cyclist(self, actor):

        return any(keyword in actor.type_id for keyword in self.CYCLIST_TYPE_KEYWORDS)

    def _update_cyclist_clearance(self, waypoint, vehicle):
        """
        Keeps a lateral gap from a cyclist ahead (HazardAtSideLane) by
        offsetting the tracked line away from the cyclist's side, without
        leaving the lane. Speed is still governed by car-following.

        Uses the waypoint's own right vector as the sign basis (the same
        basis the Stanley controller applies the offset in), so a cyclist
        on the right (lateral > 0) produces a shift to the left, and vice
        versa. Resets the offset to 0 when there is no cyclist to clear.

            :param waypoint: the agent's current waypoint
            :param vehicle: the lead obstacle detected ahead, or None
        """
        if vehicle is None or not self._is_cyclist(vehicle):
            self._local_planner.set_offset(0)
            return

        r_vec = waypoint.transform.get_right_vector()
        origin = waypoint.transform.location
        loc = vehicle.get_location()
        lateral = (loc.x - origin.x) * r_vec.x + (loc.y - origin.y) * r_vec.y

        # Positive offset shifts right; move away from the cyclist's side.
        offset = -self.CYCLIST_CLEARANCE_OFFSET if lateral > 0 else self.CYCLIST_CLEARANCE_OFFSET
        self._local_planner.set_offset(offset)

    def _static_obstacle_ahead(self, waypoint):

        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        static_list = [o for o in obstacle_list if "static.prop" in o.type_id]
        obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(
            static_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=self.FORWARD_ANGLE_STRAIGHT)
        if obstacle_state and not self._is_obstacle_in_lane(obstacle, waypoint):
            return False, None, -1
        return obstacle_state, obstacle, distance


    def _update_stall_tracking(self, vehicle):

        if self._speed > self.STALL_MIN_EGO_MOVED_SPEED_KMH:
            self._ego_has_moved_normally = True

        if vehicle is None or get_speed(vehicle) > self.STALL_SPEED_THRESHOLD:
            self._stalled_vehicle_id = None
            self._stalled_tick_counter = 0
            self._ego_blocked_tick_counter = 0
            return

        if vehicle.id != self._stalled_vehicle_id:
            self._stalled_vehicle_id = vehicle.id
            self._stalled_tick_counter = 0
            self._ego_blocked_tick_counter = 0
            return

        self._stalled_tick_counter += 1
        if self._speed <= self.EGO_BLOCKED_SPEED_THRESHOLD:
            self._ego_blocked_tick_counter += 1
        else:
            self._ego_blocked_tick_counter = 0

    def _stalled_vehicle_confirmed(self):

        if (not self._ego_has_moved_normally
                and self._ego_blocked_tick_counter <= self.STALL_STARTUP_HARD_BLOCK_TICKS):
            return False
        if self._tick_count <= self.STALL_STARTUP_GRACE_TICKS:
            return False
        return (self._stalled_tick_counter > self.STALL_TIMEOUT_TICKS
                and self._ego_blocked_tick_counter > self.EGO_BLOCKED_CONFIRM_TICKS)

    def _near_junction(self, location, distance):

        wp = self._map.get_waypoint(location)
        if wp.is_junction:
            return True
        ahead = wp.next(distance)
        return any(w.is_junction for w in ahead)

    def _stalled_vehicle_ahead(self, waypoint):

        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
            obstacle_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=self.FORWARD_ANGLE_STRAIGHT)

        if vehicle_state and self._is_cyclist(vehicle):
            vehicle_state = False

        if vehicle_state and vehicle.id in self._vehicles_seen_moving:
            self._update_stall_tracking(None)
            return False, None, -1

        if vehicle_state and (
                self._near_junction(waypoint.transform.location, self.JUNCTION_STALL_EXCLUSION_DISTANCE)
                or self._near_junction(vehicle.get_location(), self.JUNCTION_STALL_EXCLUSION_DISTANCE)):
            self._update_stall_tracking(None)
            return False, None, -1

        self._update_stall_tracking(vehicle if vehicle_state else None)
        if (vehicle_state and self._stalled_vehicle_confirmed()
                and self._is_obstacle_in_lane(vehicle, waypoint)):
            return True, vehicle, distance
        return False, None, -1


    def _blocking_obstacle_ahead(self, waypoint):
        """
        Generic entry point for the bypass module. An overtake is only
        ever justified by one of exactly two situations:
          1. a static obstacle (ConstructionObstacleTwoWays, roadworks...);
          2. a vehicle confirmed genuinely stalled - NOT a car paused at a
             stop sign/red light/pulling away (see `_stalled_vehicle_ahead`).
        Cyclists (HazardAtSideLane) are intentionally excluded: they are
        handled by car-following plus a lateral clearance offset (see
        `_update_cyclist_clearance`), never overtaken.
        Anything else in front of the agent is plain car-following
        (`car_following_manager`), never a reason to change lane.

            :param waypoint: the agent's current waypoint
            :return: tuple (obstacle_state, obstacle, distance)
        """
        static_state, static_obstacle, static_distance = self._static_obstacle_ahead(waypoint)
        if static_state:
            return static_state, static_obstacle, static_distance

        # Cyclists (HazardAtSideLane) are deliberately NOT bypassed here.
        # They are handled as a slow lead vehicle by car-following, plus a
        # lateral clearance offset (see _update_cyclist_clearance), per the
        # route plan (slow down + keep spacing, never overtake into oncoming).
        return self._stalled_vehicle_ahead(waypoint)

    def _oncoming_lane_obstacle(self, waypoint):

        actors = self._actors.filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in actors if dist(v) < self.BYPASS_DETECTION_DISTANCE and v.id != self._vehicle.id]
        return self._vehicle_obstacle_detected(
            vehicle_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=180, lane_offset=-1)

#----------------------------------------------------------------------------------------------#
# BYPASS
#----------------------------------------------------------------------------------------------#

    def _gap_is_safe(self, oncoming_distance, oncoming_speed, min_gap_time=None):

        if min_gap_time is None:
            min_gap_time = self.BYPASS_MIN_GAP_TIME

        if oncoming_distance is None or oncoming_distance < 0:
            return True

        speed_ms = oncoming_speed / 3.6
        if speed_ms <= 0:
            return True

        time_to_arrival = oncoming_distance / speed_ms
        return time_to_arrival >= min_gap_time

    def _oncoming_gap_clear(self, waypoint):
        """
        Gap acceptance before pulling into the opposite (left) lane to
        bypass an obstacle: returns True only if no oncoming vehicle is
        close enough to make the manoeuvre unsafe.

            :param waypoint: the agent's current waypoint
            :return: True if it is safe to move into the oncoming lane
        """
        oncoming_state, oncoming_vehicle, oncoming_distance = self._oncoming_lane_obstacle(waypoint)
        if not oncoming_state:
            return True
        return self._gap_is_safe(oncoming_distance, get_speed(oncoming_vehicle))

    def _obstacle_cleared(self):

        if self._bypassed_vehicle is None:
            return False

        ego_loc = self._vehicle.get_location()
        obs_loc = self._bypassed_vehicle.get_location()

        ego_wp = self._map.get_waypoint(ego_loc)
        obs_wp = self._map.get_waypoint(obs_loc)

        if ego_wp.road_id != obs_wp.road_id:
            return False

        if ego_wp.lane_id == obs_wp.lane_id:
            return False

        return ego_loc.distance(obs_loc) > 15

    def _reset_bypass_state(self):

        self._bypassing = False
        self._bypassed_vehicle = None
        self._bypass_state = 'idle'
        self._bypass_origin_waypoint = None
        self._bypass_start_tick = None
        self._last_bypass_transition_tick = None
        self._bypass_target_actor = None
        self._bypass_target_resumed_tick_counter = 0
        self._bypass_best_offset = None
        self._bypass_best_offset_tick = None
        self._stalled_vehicle_id = None
        self._stalled_tick_counter = 0
        self._ego_blocked_tick_counter = 0

    def _try_lane_change(self, direction):
        """
        Like BasicAgent.lane_change, but only commits the new plan if a valid
        path was actually found. BasicAgent.lane_change sets an empty plan when
        no path exists, which then crashes the controller (empty waypoint
        queue -> IndexError). This guard prevents that.

            :param direction: 'left' or 'right'
            :return: True if a lane-change path was found and applied
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            0,            # same-lane distance
            0,            # other-lane distance
            2 * speed,    # lane-change distance
            False,        # check
            1,            # number of lane changes
            self._sampling_resolution)
        if not path:
            return False
        self.set_global_plan(path)
        return True

    def bypass_obstacle_manager(self, waypoint):
        """
        Decide whether an overtake of a blocking obstacle should be started,
        continued, or ended.

            :param waypoint: the agent's current waypoint
            :return: True if the bypass module is handling this tick (run_step
                should defer to it), False otherwise.
        """
        if not self._bypassing:
            obstacle_state, obstacle, _ = self._blocking_obstacle_ahead(waypoint)
            if not obstacle_state:
                return False

            # The overtake uses the opposite (left) lane on a two-way road.
            left_lane = waypoint.get_left_lane()
            if left_lane is None or left_lane.lane_type != carla.LaneType.Driving:
                # No usable opposite lane: leave it to car-following/braking.
                return False

            # Gap acceptance: wait behind the obstacle until the oncoming
            # lane is clear enough, instead of pulling out into traffic.
            if not self._oncoming_gap_clear(waypoint):
                return True

            # Only commit to the manoeuvre if a lane-change path exists.
            if not self._try_lane_change("left"):
                return True

            self._log_bypass_transition('gap_found', waypoint, obstacle)
            self._bypassing = True
            self._bypassed_vehicle = obstacle
            self._bypass_start_tick = self._tick_count
            return True

        # Currently overtaking.
        if self._obstacle_cleared():
            if self._try_lane_change("right"):
                self._log_bypass_transition('success', waypoint)
                self._reset_bypass_state()
                return False
            # No return path yet: stay the course and retry next tick rather
            # than crashing on an empty plan.
            return True

        if self._bypass_timed_out():
            self._try_lane_change("right")   # best effort, ignore failure
            self._log_bypass_transition('timeout', waypoint)
            self._reset_bypass_state()
            return False

        return True

#----------------------------------------------------------------------------------------------#
#   JUNCTION
#----------------------------------------------------------------------------------------------#

    def _junction_ahead(self):

        return self._incoming_waypoint.is_junction and self._incoming_direction in (RoadOption.LEFT, RoadOption.RIGHT)

    def _cross_traffic_obstacle(self, waypoint):

        vehicle_list = self._build_obstacle_list(waypoint, max_distance=self.JUNCTION_DETECTION_DISTANCE)
        return self._vehicle_obstacle_detected(
            vehicle_list, self.JUNCTION_DETECTION_DISTANCE, up_angle_th=180)

    def _junction_gap_is_safe(self, obstacle_state, obstacle_vehicle, obstacle_distance):

        if not obstacle_state:
            return True
        return self._gap_is_safe(obstacle_distance, get_speed(obstacle_vehicle),
                                  min_gap_time=self.JUNCTION_MIN_GAP_TIME)

    def _junction_timed_out(self):

        return (self._tick_count - self._junction_start_tick) > self.JUNCTION_TIMEOUT_TICKS

    def _reset_junction_state(self):
        """Resets the status of the junction-crossing module."""
        self._junction_state = 'idle'
        self._junction_start_tick = None

    def junction_manager(self, waypoint):
        """
        Generic module for junction crossing. Drives the cycle of
        detection -> waiting for a clear gap -> crossing
        """
        if not self._junction_ahead():
            if self._junction_state != 'idle':
                self._reset_junction_state()
            return False

        if self._junction_state == 'crossing':
            return False

        if self._junction_state == 'idle':
            self._junction_state = 'waiting_clear'
            self._junction_start_tick = self._tick_count
            self._log_junction_transition('waiting', waypoint)

        if self._junction_timed_out():
            self._log_junction_transition('timeout', waypoint)
            self._reset_junction_state()
            return False

        obstacle_state, obstacle_vehicle, obstacle_distance = self._cross_traffic_obstacle(waypoint)
        if self._junction_gap_is_safe(obstacle_state, obstacle_vehicle, obstacle_distance):
            self._log_junction_transition('clear', waypoint)
            self._junction_state = 'crossing'
            return False

        return True

    def _hold_position(self, debug=False):

        self._local_planner.set_speed(0)
        return self._local_planner.run_step(debug=debug)

#----------------------------------------------------------------------------------------------#
# PEDESTRIAN AVOID
#----------------------------------------------------------------------------------------------#

    def _update_pedestrian_wait_tracking(self, walker):
        """
        Tracks how long the agent has been forced to stop for the same
        pedestrian, to tell a genuinely crossing/approaching pedestrian
        apart from one merely loitering near the road (background NPC,
        sidewalk, bus stop) that should not cause an indefinite freeze.

            :param walker: the pedestrian currently forcing an emergency
                stop, or None if the agent isn't currently stopped for one
        """
        if walker is None:
            self._pedestrian_wait_id = None
            self._pedestrian_wait_tick_counter = 0
            return

        if walker.id != self._pedestrian_wait_id:
            self._pedestrian_wait_id = walker.id
            self._pedestrian_wait_tick_counter = 0

        self._pedestrian_wait_tick_counter += 1

    def _pedestrian_wait_timed_out(self):
        """
        :return: True once the agent has been stopped for the same
            pedestrian longer than PEDESTRIAN_WAIT_TIMEOUT_TICKS (prevents
            an indefinite freeze in front of a stationary/loitering pedestrian).
        """
        return self._pedestrian_wait_tick_counter > self.PEDESTRIAN_WAIT_TIMEOUT_TICKS

    def _pedestrian_is_stationary(self, walker):

        return get_speed(walker) < self.PEDESTRIAN_STATIONARY_SPEED

    def _creep_past_pedestrian(self, debug=False):
        """
        Resumes driving at a low, cautious speed once the wait for a
        stationary pedestrian has timed out, instead of remaining fully
        stopped forever.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._local_planner.set_speed(self.PEDESTRIAN_CREEP_SPEED)
        return self._local_planner.run_step(debug=debug)

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

        walker_list = self._actors.filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < self._collision_detection_range()]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, self._dynamic_forward_distance(max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2)), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, self._dynamic_forward_distance(max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2)), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, self._collision_detection_range(), up_angle_th=60)

        return walker_state, walker, distance

#----------------------------------------------------------------------------------------------#
# CONTROL LOSS
#----------------------------------------------------------------------------------------------#

    def _road_heading(self, waypoint):
        """
        :param waypoint: the agent's current waypoint
        :return: the road's own heading (degrees) at that waypoint
        """
        return waypoint.transform.rotation.yaw

    def _velocity_heading(self):

        vel = self._vehicle.get_velocity()
        return math.degrees(math.atan2(vel.y, vel.x))

    def _heading_deviation(self, waypoint):

        diff = (self._velocity_heading() - self._road_heading(waypoint)) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        return diff

    def _wet_severity(self):

        try:
            weather = self._world.get_weather()
        except AttributeError:
            return 0.0
        wetness = getattr(weather, 'wetness', 0.0)
        precipitation = getattr(weather, 'precipitation', 0.0)
        return max(wetness, precipitation) / 100.0

    def _control_loss_heading_threshold(self):

        severity = self._wet_severity()
        margin = 1.0 - (1.0 - self.WET_HEADING_MARGIN) * severity
        return self.CONTROL_LOSS_HEADING_THRESHOLD * margin

    def _control_loss_stabilize_speed(self):

        severity = self._wet_severity()
        margin = 1.0 - (1.0 - self.WET_SPEED_MARGIN) * severity
        return self.CONTROL_LOSS_STABILIZE_SPEED * margin

    def _in_bypass_maneuver_grace_period(self):

        if not self._bypassing or self._bypass_start_tick is None:
            return False
        return (self._tick_count - self._bypass_start_tick) <= self.CONTROL_LOSS_BYPASS_GRACE_TICKS

    def _control_loss_detected(self, waypoint):

        if self._direction in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            return False
        if self._in_bypass_maneuver_grace_period():
            return False
        if self._speed < self.CONTROL_LOSS_MIN_SPEED_KMH:
            return False
        return self._heading_deviation(waypoint) > self._control_loss_heading_threshold()

    def _control_loss_recovered(self, waypoint):

        return self._heading_deviation(waypoint) < self.CONTROL_LOSS_RECOVERY_THRESHOLD

    def _control_loss_timed_out(self):

        return (self._tick_count - self._control_loss_start_tick) > self.CONTROL_LOSS_TIMEOUT_TICKS

    def _reset_control_loss_state(self):
        """Resets the status of the control-loss stabilization module."""
        self._control_loss_state = 'idle'
        self._control_loss_start_tick = None
        self._control_loss_pending_ticks = 0

    def control_loss_manager(self, waypoint):

        if self._control_loss_state == 'idle':
            if self._control_loss_detected(waypoint):
                self._control_loss_pending_ticks += 1
                if self._control_loss_pending_ticks >= self.CONTROL_LOSS_DEBOUNCE_TICKS:
                    self._control_loss_state = 'stabilizing'
                    self._control_loss_start_tick = self._tick_count
                    self._control_loss_pending_ticks = 0
                    self._log_control_loss_transition('detected', waypoint)
            else:
                self._control_loss_pending_ticks = 0
            return self._control_loss_state != 'idle'

        if self._control_loss_timed_out():
            self._log_control_loss_transition('timeout', waypoint)
            self._reset_control_loss_state()
            return False

        if self._control_loss_recovered(waypoint):
            self._log_control_loss_transition('recovered', waypoint)
            self._reset_control_loss_state()
            return False

        return True

#----------------------------------------------------------------------------------------------#
# CAR FOLLOWING
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
            self._local_planner.set_speed(self._clamp_to_speed_limit(target_speed))
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(self._clamp_to_speed_limit(target_speed))
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(self._clamp_to_speed_limit(target_speed))
            control = self._local_planner.run_step(debug=debug)

        return control

#----------------------------------------------------------------------------------------------#
#   RUN_STEP
#----------------------------------------------------------------------------------------------#

    def _clamp_to_speed_limit(self, target_speed):
        """
        Hard safety net: never request a cruise speed above the posted
        speed limit, whichever branch computed it. Every formula above
        already subtracts a margin (`speed_lim_dist`) before capping, so
        this should be a no-op in practice - it only guards against a
        branch forgetting that margin (e.g. the junction branch below uses
        a fixed "-5" instead of `speed_lim_dist`) or a transient overshoot
        requested upstream. Note this does not fix PID overshoot inside
        the longitudinal controller itself (controller.py) - that would
        need its own tuning, out of scope here.

            :param target_speed: the cruise speed a branch wants to request
            :return: target_speed, capped at the current speed limit
        """
        return min(target_speed, self._speed_limit)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()
        self._tick_count += 1

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        self._log_vehicle_state(ego_vehicle_wp)

        # 1: Red lights behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 1b: Stop-sign compliance
        if self.stop_sign_manager():
            return self._hold_position(debug=debug)

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close (distance threshold
            # scales with current speed, see _effective_braking_distance).
            if distance < self._effective_braking_distance():
                self._update_pedestrian_wait_tracking(walker)
                if self._pedestrian_wait_timed_out() and self._pedestrian_is_stationary(walker):
                    self._log_pedestrian_wait_timeout(ego_vehicle_wp)
                    blocking_control = self._react_to_moving_obstacle(ego_vehicle_wp, debug=debug)
                    if blocking_control is not None:
                        return blocking_control
                    return self._creep_past_pedestrian(debug=debug)
                return self.emergency_stop()
            self._update_pedestrian_wait_tracking(None)
        else:
            self._update_pedestrian_wait_tracking(None)

        if self.control_loss_manager(ego_vehicle_wp):
            target_speed = min([
                self._control_loss_stabilize_speed(),
                self._behavior.max_speed])
            self._local_planner.set_speed(self._clamp_to_speed_limit(target_speed))
            return self._local_planner.run_step(debug=debug)

        # 2.2: Static obstacle bypass behavior (construction/accident/parked vehicle)
        if self.bypass_obstacle_manager(ego_vehicle_wp):
            # The obstacle being overtaken is ignored inside
            # _react_to_moving_obstacle via _should_ignore_vehicle.
            blocking_control = self._react_to_moving_obstacle(ego_vehicle_wp, debug=debug)
            if blocking_control is not None:
                return blocking_control

            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(self._clamp_to_speed_limit(target_speed))
            return self._local_planner.run_step(debug=debug)

        # 2.3: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        # Keep a lateral gap from a cyclist ahead without changing lane.
        self._update_cyclist_clearance(ego_vehicle_wp, vehicle if vehicle_state else None)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close (distance threshold
            # scales with current speed, see _effective_braking_distance).
            if distance < self._effective_braking_distance():
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior (BlockedIntersection / NonSignalizedJunctionRightTurn)
        elif self._junction_ahead():
            if self.junction_manager(ego_vehicle_wp):
                control = self._hold_position(debug=debug)
            else:
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - 5])
                self._local_planner.set_speed(self._clamp_to_speed_limit(target_speed))
                control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(self._clamp_to_speed_limit(target_speed))
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