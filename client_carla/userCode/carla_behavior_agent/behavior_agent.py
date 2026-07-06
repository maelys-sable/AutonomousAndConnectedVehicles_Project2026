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
    BYPASS_MANEUVER_SPEED = 25      # km/h cap while alongside the obstacle (props/parked vehicles are close)
    BYPASS_FORWARD_MARGIN = 8       # m, extra stopping margin kept ahead during the manoeuvre

    STATE_LOG_INTERVAL = 20
    BYPASS_TIMEOUT_TICKS = 800

    # Junction crossing (Block 3: BlockedIntersection / NonSignalizedJunctionRightTurn)
    JUNCTION_DETECTION_DISTANCE = 30
    JUNCTION_MIN_GAP_TIME = 3.0
    JUNCTION_TIMEOUT_TICKS = 300

    # Stalled-vehicle detection (AccidentTwoWays with a wrecked vehicle rather
    # than a static.prop): a vehicle ahead reporting near-zero speed for a
    # sustained period is treated as a permanent obstacle to bypass, the same
    # way a construction cone or a parked car already are.
    STALL_SPEED_THRESHOLD = 1.0    # km/h, considered "not moving"
    STALL_TIMEOUT_TICKS = 150      # ~7.5s at 20 FPS before it's confirmed stalled
                                    # (long enough not to mistake a stop-sign/queue pause for a wreck)

    # Pedestrian wait (non-blocage indéfini): a loitering pedestrian near the
    # road (background NPC, not a scripted scenario) can otherwise force an
    # unconditional, permanent emergency_stop with no re-evaluation.
    PEDESTRIAN_WAIT_TIMEOUT_TICKS = 200   # ~10s at 20 FPS before creeping past
    PEDESTRIAN_STATIONARY_SPEED = 0.5     # km/h, considered "not walking"
    PEDESTRIAN_CREEP_SPEED = 5            # km/h, cautious speed once timed out

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
        self._bypass_tick_counter = 0
        self._pre_bypass_behavior = None  # saved profile while the Cautious swap is active

        self._actors = None

        self._tick_count = 0
        self._scenario_result = None

        # Cycle : 'idle' -> 'waiting_clear' -> 'crossing' -> 'idle'
        self._junction_state = 'idle'
        self._junction_tick_counter = 0

        # Stalled-vehicle tracking (see STALL_TIMEOUT_TICKS)
        self._stalled_vehicle_id = None
        self._stalled_tick_counter = 0

        # Pedestrian wait tracking (see PEDESTRIAN_WAIT_TIMEOUT_TICKS)
        self._pedestrian_wait_id = None
        self._pedestrian_wait_tick_counter = 0

    def _refresh_actor_snapshot(self):
        """
        Retrieves all actors in the scene just once per tick.

        `world.get_actors()` is an RPC call to the simulator, not a simple local read. 
        Calling it separately in each manager (pedestrians, vehicles, routing, etc.) 
        unnecessarily multiplies the number of network round trips per tick; with heavy 
        traffic, this may be enough to trigger the simulator’s watchdog.
        """
        self._actors = self._world.get_actors()

    def _log_vehicle_state(self, waypoint):
        """
        Periodically displays the position and velocity of the ego vehicle,
        so that the course of a test can be replayed without restarting the simulation.

            :param waypoint: the agent’s current waypoint
        """
        if self._tick_count % self.STATE_LOG_INTERVAL != 0:
            return
        loc = waypoint.transform.location
        print(f"[STATE] tick={self._tick_count} pos=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) "
              f"speed={self._speed:.1f} km/h")

    def _log_bypass_transition(self, event, waypoint=None):
        """
        A single entry point for all messages relating to the obstacle-avoidance cycle, 
        to ensure a consistent format and to make it easy to grep the logs 
        (e.g. test success/failure).

            :param event: 'detected' | 'gap_found' | 'success' | 'timeout'
            :param waypoint: the agent’s current waypoint (optional)
        """
        loc = waypoint.transform.location if waypoint is not None else None
        pos = f" pos=({loc.x:.1f}, {loc.y:.1f})" if loc is not None else ""
        messages = {
            'detected': f"[BYPASS] Obstacle detected{pos} -> searching for a gap",
            'gap_found': f"[BYPASS] Gap found{pos} -> start of manoeuvre",
            'success': f"[BYPASS] TEST SUCCESSFUL: obstacle bypassed, back on track{pos}",
            'timeout': f"[BYPASS] TEST FAILED: manoeuvre timed out{pos}"
                       f"(stuck in '{self._bypass_state}')",
            'aborted': f"[BYPASS] TEST FAILED: no usable opposite lane{pos} -> aborting manoeuvre",
        }
        print(messages[event])
        if event == 'success':
            self._scenario_result = True
        elif event in ('timeout', 'aborted'):
            self._scenario_result = False

    def _bypass_timed_out(self):
        """
        Increments the bypass lock counter and indicates whether the operation
        has been running for too long (enables detection of a scenario failure
        without waiting for the route’s global timeout).

            :return: True if the operation is considered to be blocked
        """
        self._bypass_tick_counter += 1
        return self._bypass_tick_counter > self.BYPASS_TIMEOUT_TICKS

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._refresh_actor_snapshot()
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
        actor_list = self._actors
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
        actors = self._actors
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

    def _update_stall_tracking(self, vehicle):
        """
        Tracks how long a specific vehicle ahead has been reporting
        near-zero speed. Distinguishes a genuinely stalled/wrecked vehicle
        (AccidentTwoWays) from one only briefly stopped (queue, stop sign).

            :param vehicle: the vehicle currently detected ahead, or None
        """
        if vehicle is None or get_speed(vehicle) > self.STALL_SPEED_THRESHOLD:
            self._stalled_vehicle_id = None
            self._stalled_tick_counter = 0
            return

        if vehicle.id != self._stalled_vehicle_id:
            self._stalled_vehicle_id = vehicle.id
            self._stalled_tick_counter = 0

        self._stalled_tick_counter += 1

    def _stalled_vehicle_confirmed(self):
        """
        :return: True once the currently tracked vehicle has been
            stationary long enough to be treated as a permanent obstacle.
        """
        return self._stalled_tick_counter > self.STALL_TIMEOUT_TICKS

    def _stalled_vehicle_ahead(self, waypoint):
        """
        Detects a vehicle ahead that has been confirmed stalled (see
        `_update_stall_tracking`): a wrecked or broken-down vehicle blocking
        the lane (AccidentTwoWays), as opposed to a static prop already
        covered by `_static_obstacle_ahead`.

            :param waypoint: the agent's current waypoint
            :return: tuple (obstacle_state, obstacle, distance)
        """
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
            obstacle_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=self.FORWARD_ANGLE_STRAIGHT)

        self._update_stall_tracking(vehicle if vehicle_state else None)
        if vehicle_state and self._stalled_vehicle_confirmed():
            return True, vehicle, distance
        return False, None, -1

    def _blocking_obstacle_ahead(self, waypoint):
        """
        Generic entry point for the bypass module: an obstacle to go around
        is either a static prop (ConstructionObstacleTwoWays, roadworks...)
        or a vehicle confirmed stalled long enough to be treated the same
        way (AccidentTwoWays with a wrecked vehicle, a broken-down car...).

            :param waypoint: the agent's current waypoint
            :return: tuple (obstacle_state, obstacle, distance)
        """
        static_state, static_obstacle, static_distance = self._static_obstacle_ahead(waypoint)
        if static_state:
            return static_state, static_obstacle, static_distance
        return self._stalled_vehicle_ahead(waypoint)

    def _oncoming_lane_obstacle(self, waypoint):
        """
        Detects a vehicle travelling in the opposite direction on the opposite carriageway,
        used to search for a gap before overtaking.

            :param waypoint: the agent’s current waypoint
            :return: tuple (vehicle_state, vehicle, distance)
        """
        actors = self._actors.filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in actors if dist(v) < self.BYPASS_DETECTION_DISTANCE and v.id != self._vehicle.id]
        return self._vehicle_obstacle_detected(
            vehicle_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=180, lane_offset=-1)

    def _gap_is_safe(self, oncoming_distance, oncoming_speed, min_gap_time=None):
        """
        Assesses whether there is a sufficient gap in oncoming traffic to
        merge (gap detection / gap acceptance).

            :param oncoming_distance: distance to the oncoming vehicle (m), None/-1 if not available
            :param oncoming_speed: speed of the oncoming vehicle (km/h)
            :param min_gap_time: minimum time-to-arrival required for the gap
                to be considered safe. Defaults to BYPASS_MIN_GAP_TIME, but
                callers with a different gap-acceptance threshold (e.g. the
                junction manager) can override it.
            :return: True if the gap is deemed safe
        """
        if min_gap_time is None:
            min_gap_time = self.BYPASS_MIN_GAP_TIME

        if oncoming_distance is None or oncoming_distance < 0:
            return True  # no vehiclle detected : free line

        speed_ms = oncoming_speed / 3.6
        if speed_ms <= 0:
            return True  # vehicle stopped : no imminent risk of frontal collision

        time_to_arrival = oncoming_distance / speed_ms
        return time_to_arrival >= min_gap_time

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

    def _bypass_target_waypoints(self, waypoint):
        """
        Resolves the two waypoints needed to start a bypass: the opposite
        lane to move into, and the current local-planner target to resume
        towards afterwards. Either can legitimately be missing (edge of the
        map, lane ending, planner between targets) -- this must be checked
        before use, not assumed.

            :param waypoint: the agent's current waypoint
            :return: tuple (opposite_wpt, end_waypoint), either may be None
        """
        return waypoint.get_left_lane(), self._local_planner.target_waypoint

    def _start_bypass_maneuver(self, waypoint):
        """
        Triggers a lateral shift to the opposite lane to bypass the obstacle 
        (same mechanism as `_tailgating`: redefine the local destination to the adjacent lane).

            :param waypoint: the agent’s current waypoint
            :return: True if the manoeuvre was actually started
        """
        opposite_wpt, end_waypoint = self._bypass_target_waypoints(waypoint)
        if opposite_wpt is None or end_waypoint is None:
            return False
        self.set_destination(end_waypoint.transform.location, opposite_wpt.transform.location)
        return True

    def _obstacle_cleared(self, waypoint):
        """
        Indicates whether the obstacle that was bypassed has now been passed 
        (no longer detected in front of the agent).

            :param waypoint: the agent’s current waypoint
            :return: True if the obstacle is no longer a frontal obstacle
        """
        obstacle_state, _, _ = self._blocking_obstacle_ahead(waypoint)
        return not obstacle_state

    def _back_on_original_lane(self, waypoint):
        """
        Indicates whether the agent has returned to its original path after taking a detour.

            :param waypoint: the agent’s current waypoint
            :return: True if the current path matches the starting path
        """
        return (self._bypass_origin_waypoint is not None
                and waypoint.lane_id == self._bypass_origin_waypoint.lane_id)

    def _enter_bypass_caution(self):
        """
        Switches the active behavior profile to Cautious for the duration of
        the manoeuvre (larger min_proximity_threshold and braking_distance),
        since a bypass happens right next to the obstacle being avoided.
        No-op if already Cautious (e.g. the agent was configured that way).
        """
        if not isinstance(self._behavior, Cautious):
            self._pre_bypass_behavior = self._behavior
            self._behavior = Cautious()
            print("[BYPASS] Switching to Cautious profile for the manoeuvre")

    def _exit_bypass_caution(self):
        """Restores the behavior profile saved by `_enter_bypass_caution`."""
        if self._pre_bypass_behavior is not None:
            self._behavior = self._pre_bypass_behavior
            self._pre_bypass_behavior = None
            print("[BYPASS] Restoring previous behavior profile")

    def _bypass_target_speed(self):
        """
        Conservative cruise speed while alongside the obstacle: static props
        and parked/stalled vehicles leave little lateral margin, so the
        manoeuvre is capped well below the profile's usual max_speed.

            :return: target speed in km/h
        """
        return min(self.BYPASS_MANEUVER_SPEED, self._speed_limit - self._behavior.speed_lim_dist)

    def _bypass_forward_clear(self, waypoint):
        """
        Checks for a vehicle immediately ahead in the lane currently used
        during the manoeuvre (the obstacle itself, or another vehicle
        merging back into). Without this check the bypass branch of
        `run_step` would drive blindly at `_bypass_target_speed`.

            :param waypoint: the agent's current waypoint
            :return: True if it is safe to keep driving, False if it must brake
        """
        vehicle_list = self._build_obstacle_list(waypoint, max_distance=self.OBSTACLE_MAX_DISTANCE)
        vehicle_state, vehicle, distance = self._forward_obstacle_detected(vehicle_list)
        if not vehicle_state:
            return True
        margin = max(vehicle.bounding_box.extent.x, vehicle.bounding_box.extent.y)
        return (distance - margin) >= self.BYPASS_FORWARD_MARGIN

    def _reset_bypass_state(self):
        """Resets the status of the bypass module."""
        self._bypass_state = 'idle'
        self._bypass_origin_waypoint = None
        self._bypass_tick_counter = 0
        self._stalled_vehicle_id = None
        self._stalled_tick_counter = 0
        self._exit_bypass_caution()

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
            obstacle_state, _, _ = self._blocking_obstacle_ahead(waypoint)
            if obstacle_state:
                self._bypass_origin_waypoint = waypoint
                self._bypass_state = 'waiting_gap'
                self._enter_bypass_caution()
                self._log_bypass_transition('detected', waypoint)
            return self._bypass_state != 'idle'

        if self._bypass_timed_out():
            self._log_bypass_transition('timeout', waypoint)
            self._reset_bypass_state()
            return False

        if self._bypass_state == 'waiting_gap':
            if self._can_start_bypass(waypoint):
                if not self._start_bypass_maneuver(waypoint):
                    self._log_bypass_transition('aborted', waypoint)
                    self._reset_bypass_state()
                    return False
                self._bypass_state = 'overtaking'
                self._log_bypass_transition('gap_found', waypoint)
            return True

        if self._bypass_state == 'overtaking':
            if self._obstacle_cleared(waypoint):
                self._bypass_state = 'returning'
            return True

        if self._bypass_state == 'returning':
            if self._back_on_original_lane(waypoint):
                self._log_bypass_transition('success', waypoint)
                self._reset_bypass_state()
                return False
            return True

        return False

#----------------------------------------------------------------------------------------------#

    def _junction_ahead(self):
        """
        Indicates whether the incoming waypoint is a junction the agent is
        about to enter by turning (left or right), as opposed to simply
        driving straight through an intersection.

            :return: True if a turning manoeuvre through a junction is imminent
        """
        return self._incoming_waypoint.is_junction and self._incoming_direction in (RoadOption.LEFT, RoadOption.RIGHT)

    def _cross_traffic_obstacle(self, waypoint):
        """
        Detects a vehicle blocking or crossing the junction ahead: either a
        vehicle stuck in the intersection (BlockedIntersection) or one from
        the transversal traffic flow (NonSignalizedJunctionRightTurn). Both
        scenarios share the same detection mechanism.

            :param waypoint: the agent's current waypoint
            :return: tuple (vehicle_state, vehicle, distance)
        """
        vehicle_list = self._build_obstacle_list(waypoint, max_distance=self.JUNCTION_DETECTION_DISTANCE)
        return self._vehicle_obstacle_detected(
            vehicle_list, self.JUNCTION_DETECTION_DISTANCE, up_angle_th=180)

    def _junction_gap_is_safe(self, obstacle_state, obstacle_vehicle, obstacle_distance):
        """
        Assesses whether it is safe to enter/cross the junction: no obstacle
        at all, a stopped one, or one far/slow enough to leave a large
        enough gap. Reuses the bypass module's gap-acceptance logic with a
        junction-specific minimum gap time.

            :param obstacle_state: True if a vehicle is detected at the junction
            :param obstacle_vehicle: the detected vehicle, if any
            :param obstacle_distance: distance to the detected vehicle
            :return: True if the agent may proceed
        """
        if not obstacle_state:
            return True
        return self._gap_is_safe(obstacle_distance, get_speed(obstacle_vehicle),
                                  min_gap_time=self.JUNCTION_MIN_GAP_TIME)

    def _junction_timed_out(self):
        """
        Increments the junction-wait counter and indicates whether the agent
        has been waiting too long to enter/cross (prevents the "stuck
        forever at an intersection" failure mode and the resulting scenario
        timeout infraction).

            :return: True if the wait is considered excessive
        """
        self._junction_tick_counter += 1
        return self._junction_tick_counter > self.JUNCTION_TIMEOUT_TICKS

    def _reset_junction_state(self):
        """Resets the status of the junction-crossing module."""
        self._junction_state = 'idle'
        self._junction_tick_counter = 0

    def _log_junction_transition(self, event, waypoint=None):
        """
        Single entry point for all junction-crossing log messages, mirroring
        `_log_bypass_transition` so both mechanisms stay easy to tell apart
        in the logs.

            :param event: 'waiting' | 'clear' | 'timeout'
            :param waypoint: the agent's current waypoint (optional)
        """
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

    def junction_manager(self, waypoint):
        """
        Generic module for junction crossing. Drives the cycle of
        detection -> waiting for a clear gap -> crossing, with periodic
        re-evaluation and a timeout so the agent never waits indefinitely.

            :param waypoint: the agent's current waypoint
            :return: True if the agent must hold its position (caller should
                     apply a soft stop instead of driving through the junction)
        """
        if not self._junction_ahead():
            if self._junction_state != 'idle':
                self._reset_junction_state()
            return False

        if self._junction_state == 'crossing':
            # Already committed to the manoeuvre: let the local planner
            # finish driving through the junction.
            return False

        if self._junction_state == 'idle':
            self._junction_state = 'waiting_clear'
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
        """
        Brings the vehicle to a gentle stop while waiting for a safe gap
        (junction gap acceptance), as opposed to `emergency_stop` which is
        reserved for imminent-collision braking.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._local_planner.set_speed(0)
        return self._local_planner.run_step(debug=debug)

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

        walker_list = self._actors.filter("*walker.pedestrian*")
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
        """
        Indicates whether the pedestrian forcing the stop is essentially
        not moving (loitering) rather than actively walking, which is what
        makes a cautious creep-past reasonable once the wait has timed out.

            :param walker: the pedestrian actor
            :return: True if its speed is below PEDESTRIAN_STATIONARY_SPEED
        """
        return get_speed(walker) < self.PEDESTRIAN_STATIONARY_SPEED

    def _log_pedestrian_wait_timeout(self, waypoint=None):
        """
        Logs the decision to creep past a pedestrian confirmed stationary
        after waiting too long, mirroring `_log_bypass_transition` and
        `_log_junction_transition` so all three "non-blocage indéfini"
        mechanisms stay equally visible and easy to grep in the logs.

            :param waypoint: the agent's current waypoint (optional)
        """
        loc = waypoint.transform.location if waypoint is not None else None
        pos = f" pos=({loc.x:.1f}, {loc.y:.1f})" if loc is not None else ""
        print(f"[PEDESTRIAN] Waited too long for a stationary pedestrian{pos} -> creeping past cautiously")

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

#----------------------------------------------------------------------------------------------#

    def _bypass_drive_control(self, waypoint, debug=False):
        """
        Driving control while a bypass manoeuvre is in progress. Replaces a
        blind "drive at max_speed" with a capped speed and a forward check,
        since the previous version skipped collision avoidance entirely
        while alongside the obstacle (root cause of colliding with static
        props and with a vehicle straight ahead during the manoeuvre).

            :param waypoint: the agent's current waypoint
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        if not self._bypass_forward_clear(waypoint):
            return self.emergency_stop()
        self._local_planner.set_speed(self._bypass_target_speed())
        return self._local_planner.run_step(debug=debug)

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
        self._tick_count += 1

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        self._log_vehicle_state(ego_vehicle_wp)

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
                self._update_pedestrian_wait_tracking(walker)
                if self._pedestrian_wait_timed_out() and self._pedestrian_is_stationary(walker):
                    self._log_pedestrian_wait_timeout(ego_vehicle_wp)
                    return self._creep_past_pedestrian(debug=debug)
                return self.emergency_stop()
            self._update_pedestrian_wait_tracking(None)
        else:
            self._update_pedestrian_wait_tracking(None)

        # 2.2: Static obstacle bypass behavior (construction/accident/parked vehicle)
        if self.bypass_obstacle_manager(ego_vehicle_wp):
            return self._bypass_drive_control(ego_vehicle_wp, debug=debug)

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

        # 3: Intersection behavior (BlockedIntersection / NonSignalizedJunctionRightTurn)
        elif self._junction_ahead():
            if self.junction_manager(ego_vehicle_wp):
                control = self._hold_position(debug=debug)
            else:
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