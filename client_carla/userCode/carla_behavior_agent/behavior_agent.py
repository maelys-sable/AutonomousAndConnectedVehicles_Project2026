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

    # ------------------------------------------------------------------ #
    # CARTE D'UTILISATION -- point d'entree unique : run_step().         #
    # Chaque bloc de run_step() delegue a UN manager public ; tout le    #
    # reste de la classe est un helper prive utilise UNIQUEMENT par le   #
    # manager cite. Pour savoir si un helper est mort, chercher son nom  #
    # uniquement dans la section de son manager.                        #
    #                                                                    #
    #   run_step()                                                      #
    #     1. traffic_light_manager()                                    #
    #     2.1 pedestrian_avoid_manager() + _update_pedestrian_wait_*,    #
    #         _pedestrian_wait_timed_out, _pedestrian_is_stationary,     #
    #         _log_pedestrian_wait_timeout, _creep_past_pedestrian       #
    #     2.2 bypass_obstacle_manager()  (voir sous-carte plus bas)      #
    #     2.3 collision_and_car_avoid_manager() + car_following_manager, #
    #         _tailgating, _build_obstacle_list, _forward_obstacle_*,    #
    #         _lane_change_obstacle_detected, _forward_detection_angle,  #
    #         _is_in_turn                                                #
    #     3.  junction_manager() + _junction_ahead, _cross_traffic_*,    #
    #         _junction_gap_is_safe, _junction_timed_out,                #
    #         _reset_junction_state, _log_junction_transition,           #
    #         _hold_position                                             #
    #                                                                    #
    #   bypass_obstacle_manager() sous-carte (etats: idle -> nudging OU  #
    #   waiting_gap -> overtaking -> returning -> idle) :                #
    #     detection      : _blocking_obstacle_ahead (-> _static_obstacle_#
    #                      ahead, _stalled_vehicle_ahead + _update_stall_#
    #                      tracking, _stalled_vehicle_confirmed),        #
    #                      _obstacles_ahead, _farthest_obstacle_distance #
    #     dimensionnement : _widest_bypass_offset, _lane_max_offset,     #
    #                      _obstacle_side_clearance,                     #
    #                      _obstacle_lateral_offset, _set_lane_offset    #
    #     nudging        : _bypass_refresh_nudge, _fallback_to_full_     #
    #                      bypass, _nudge_convergence_speed              #
    #     gap acceptance : _can_start_bypass, _bypass_side,              #
    #                      _right_lane_usable, _right_lane_clear,        #
    #                      _oncoming_lane_obstacle, _vehicles_on_lane,   #
    #                      _gap_is_safe                                 #
    #     manoeuvre      : _start_bypass_maneuver, _build_lane_shift_path, #
    #                      _bypass_remaining_distance, _forward_on_lane,   #
    #                      _resume_original_lane, _overtaking_transition_  #
    #                      speed                                          #
    #     sortie         : _obstacle_cleared, _bypass_progress_clear,    #
    #                      _back_on_original_lane, _bypass_timed_out,    #
    #                      _reset_bypass_state                          #
    #     profil/vitesse : _enter_bypass_caution, _exit_bypass_caution,  #
    #                      _bypass_target_speed, _bypass_convergence_    #
    #                      speed, _nudge_convergence_speed,               #
    #                      _overtaking_transition_speed,                  #
    #                      _waiting_gap_speed, _bypass_drive_control,     #
    #                      _bypass_forward_clear                        #
    #     logs           : _log_bypass_transition                       #
    #     diagnostic (hors etats, appelable a tout moment) :             #
    #                      bypass_diagnostics, format_bypass_            #
    #                      diagnostics, log_bypass_diagnostics,          #
    #                      BYPASS_DEBUG                                 #
    #                                                                    #
    #   Appeles a chaque tick, hors arbre ci-dessus :                    #
    #     _update_information, _refresh_actor_snapshot,                  #
    #     _log_vehicle_state, emergency_stop                            #
    # ------------------------------------------------------------------ #

    OBSTACLE_MAX_DISTANCE = 45
    FORWARD_ANGLE_STRAIGHT = 30
    FORWARD_ANGLE_TURN = 60

    BYPASS_DETECTION_DISTANCE = 80
    BYPASS_MIN_GAP_TIME = 4.0
    BYPASS_MANEUVER_SPEED = 25      # km/h cap while alongside the obstacle (props/parked vehicles are close)
    BYPASS_FORWARD_MARGIN = 8       # m, extra stopping margin kept ahead during the manoeuvre
    BYPASS_CLEAR_MARGIN = 8         # m, extra distance past the farthest prop of the group before merging back
    BYPASS_RESUME_DISTANCE = 40     # m ahead on the original lane once the group is cleared
    BYPASS_OFFSET_MARGIN = 0.4      # m, safety margin kept from the lane edge when nudging in-lane
    BYPASS_OFFSET_CLEARANCE = 0.5   # m, extra clearance wanted past each obstacle's own edge
    BYPASS_MIN_MANEUVER_SPEED = 15  # km/h floor for a nudge that needs most of the lane's width
    BYPASS_TRANSITION_DISTANCE = 25 # m, distance from the manoeuvre's origin during which the
                                     # agent is still executing the sharp lateral shift that
                                     # GlobalRoutePlanner.trace_route produces for a lane-change
                                     # edge: it jumps straight from the current waypoint to a
                                     # point several samples into the target lane, with NO
                                     # smoothing waypoints in between (unlike an in-lane nudge,
                                     # which ramps the offset gradually). Left uncapped, the
                                     # Stanley controller was still mid-drift toward the target
                                     # lane -- short of the width it needed -- by the time the
                                     # agent reached the obstacle, clipping it (see the collision
                                     # right after "Gap found -> start of manoeuvre" in the logs).
    BYPASS_WAITING_STOP_DISTANCE = 20  # m, distance to the tracked obstacle at which the agent
                                        # must be fully stopped if no gap has opened yet. Per the
                                        # strategy doc (ConstructionObstacleTwoWays/ParkedObstacleTwoWays
                                        # /AccidentTwoWays -- "ralentir et attendre une fenetre de
                                        # degagement avant de contourner"): while 'waiting_gap',
                                        # the agent used to cruise at the manoeuvre's normal speed
                                        # the entire time no gap was found, since nothing capped
                                        # its speed in that state -- only `_bypass_forward_clear`'s
                                        # BYPASS_FORWARD_MARGIN (8 m) last-resort emergency brake
                                        # eventually fired, far too late to stop from ~25 km/h (see
                                        # the log: no "Gap found" message ever appears before the
                                        # collision). This constant sizes a deceleration ramp so the
                                        # agent comes to a controlled stop well before that.

    STATE_LOG_INTERVAL = 20
    BYPASS_TIMEOUT_TICKS = 800
    BYPASS_DEBUG = True   # opt-in: prints bypass_diagnostics every tick a
                          # manoeuvre is active (see log_bypass_diagnostics),
                          # instead of adding print statements by hand each
                          # time the raw numbers behind a decision are needed.
                          # Set back to False once the current investigation
                          # is done, to keep the logs quiet again.

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
        self._bypass_reach = 0.0  # farthest tracked obstacle's distance, captured once at detection
        self._bypass_cluster_ids = frozenset()  # ids to ignore in _bypass_forward_clear, see below

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
            'nudging': f"[BYPASS] In-lane nudge{pos} -> clearing the obstacle without a lane change",
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

    def _bypass_range_obstacle_detected(self, obstacle_list):
        """
        Shared `_vehicle_obstacle_detected` call used by both
        `_static_obstacle_ahead` and `_stalled_vehicle_ahead` -- the two
        only ever differed by which list they passed in, not by any of
        the detection parameters.

            :param obstacle_list: candidates to check (static props or vehicles)
            :return: tuple (obstacle_state, obstacle, distance)
        """
        return self._vehicle_obstacle_detected(
            obstacle_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=self.FORWARD_ANGLE_STRAIGHT)

    def _static_obstacle_ahead(self, waypoint):
        """
        Detects a static obstacle (roadworks, accident, parked vehicle)
        blocking the agent’s path, within the detour range.

            :param waypoint: the agent’s current waypoint
            :return: tuple (obstacle_state, obstacle, distance)
        """
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        static_list = [o for o in obstacle_list if "static.prop" in o.type_id]
        return self._bypass_range_obstacle_detected(static_list)

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
        vehicle_state, vehicle, distance = self._bypass_range_obstacle_detected(obstacle_list)

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

    def _vehicles_on_lane(self, target_wpt, max_distance):
        """
        Vehicles actually located on `target_wpt`'s lane, matched by real
        `road_id`/`lane_id` -- NOT by `_vehicle_obstacle_detected`'s
        `lane_offset` arithmetic (`ego_wpt.lane_id + lane_offset`), which
        assumes the target lane is a same-direction neighbour reached by
        a small integer step (e.g. lane -1 -> -2 for a right lane change).
        That assumption holds for a same-direction lane change, but not
        for crossing to the OPPOSITE lane on a two-way road: the opposite
        lane's id is on the other side of the sign boundary (e.g. -1 -> +1,
        skipping 0 entirely), so `ego_wpt.lane_id - 1` never identifies it.
        The agent then found no vehicle there regardless of what was
        actually on that lane and treated every gap as safe -- which is
        why it collided with an oncoming vehicle right after "Gap found":
        the vehicle was never actually seen.

            :param target_wpt: waypoint on the lane to check, or None
            :param max_distance: search radius from the ego (m)
            :return: list of (distance, vehicle) tuples, nearest first
        """
        if target_wpt is None:
            return []
        ego_loc = self._vehicle.get_location()
        found = []
        for actor in self._actors.filter("*vehicle*"):
            if actor.id == self._vehicle.id:
                continue
            actor_wpt = self._map.get_waypoint(actor.get_location(), lane_type=carla.LaneType.Any)
            if actor_wpt is None:
                continue
            if actor_wpt.road_id != target_wpt.road_id or actor_wpt.lane_id != target_wpt.lane_id:
                continue
            distance = actor.get_location().distance(ego_loc)
            if distance < max_distance:
                found.append((distance, actor))
        found.sort(key=lambda item: item[0])
        return found

    def _oncoming_lane_obstacle(self, waypoint):
        """
        Detects the nearest vehicle travelling on the opposite/oncoming
        lane (the one used for a left-side bypass), for gap acceptance
        before crossing. See `_vehicles_on_lane` for why this matches by
        the opposite lane's real id instead of an ego-relative offset.

            :param waypoint: the agent's current waypoint
            :return: tuple (vehicle_state, vehicle, distance)
        """
        opposite_wpt = waypoint.get_left_lane()
        found = self._vehicles_on_lane(opposite_wpt, self.BYPASS_DETECTION_DISTANCE)
        if not found:
            return False, None, -1
        distance, vehicle = found[0]
        return True, vehicle, distance

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

    def _right_lane_usable(self, waypoint):
        """
        Right-hand lane candidate for a bypass that stays on the correct
        side of the road, i.e. a same-direction driving lane -- as opposed
        to the opposite (oncoming) lane used as a fallback. Prefer this
        side whenever possible: it never requires a gap in oncoming
        traffic, unlike a bypass via the left lane.

            :param waypoint: the agent's current waypoint
            :return: the right-lane waypoint if it qualifies, else None
        """
        right_wpt = waypoint.get_right_lane()
        if right_wpt is None or right_wpt.lane_type != carla.LaneType.Driving:
            return None
        if waypoint.lane_id * right_wpt.lane_id <= 0:
            return None  # sign flip -> that "right lane" is oncoming traffic
        return right_wpt

    def _right_lane_clear(self, waypoint):
        """
        Checks whether the right-hand lane is free over the bypass
        detection range, so it can actually be used as an overtaking path.
        Uses `_vehicles_on_lane` (real road_id/lane_id match) rather than
        `_vehicle_obstacle_detected`'s ego-relative `lane_offset` for the
        same reason as `_oncoming_lane_obstacle`: consistent, and safe
        even on a road with more than one lane per direction where
        `ego_wpt.lane_id + 1` might not be the specific lane
        `_right_lane_usable` actually resolved.

            :param waypoint: the agent's current waypoint
            :return: True if no vehicle occupies the right lane
        """
        right_wpt = self._right_lane_usable(waypoint)
        return not self._vehicles_on_lane(right_wpt, self.BYPASS_DETECTION_DISTANCE)

    def _bypass_side(self, waypoint):
        """
        Picks which side a full lane-level bypass should use: the right
        lane whenever it exists, is a same-direction lane, and is clear
        (no oncoming-traffic risk), falling back to the opposite (left)
        lane otherwise -- the only option on a plain two-way road.

            :param waypoint: the agent's current waypoint
            :return: 1 to bypass via the right lane, -1 via the left lane
        """
        if self._right_lane_usable(waypoint) is not None and self._right_lane_clear(waypoint):
            return 1
        return -1

    def _can_start_bypass(self, waypoint):
        """
        Determines whether the agent can move onto the chosen lane to go
        round the detected obstacle. A right-lane bypass (see
        `_bypass_side`) needs no oncoming-traffic check: it's already
        confirmed clear by `_right_lane_clear`. Otherwise, the left/
        opposite lane requires a safe gap in oncoming traffic.

            :param waypoint: the agent’s current waypoint
            :return: True if the manoeuvre can begin
        """
        if self._bypass_side(waypoint) == 1:
            return True
        oncoming_state, oncoming_vehicle, oncoming_distance = self._oncoming_lane_obstacle(waypoint)
        if not oncoming_state:
            return True
        return self._gap_is_safe(oncoming_distance, get_speed(oncoming_vehicle))

    def _is_ahead_of_vehicle(self, waypoint, obstacle):
        """
        True if `obstacle` lies ahead of the agent's current heading
        rather than behind it. Used by `_obstacles_ahead` so a prop
        just cleared -- very close, but now behind -- can't be mistaken
        for one still to be dealt with.

            :param waypoint: the agent's current waypoint
            :param obstacle: candidate static prop / vehicle
            :return: True if the obstacle is in front of the agent
        """
        forward = waypoint.transform.get_forward_vector()
        wp_loc = waypoint.transform.location
        obs_loc = obstacle.get_location()
        dx, dy = obs_loc.x - wp_loc.x, obs_loc.y - wp_loc.y
        return dx * forward.x + dy * forward.y > 0

    def _obstacles_ahead(self, waypoint, max_distance=None):
        """
        Every static obstacle ahead of the agent within a search radius,
        nearest first. Replaces the old distance-chained "cluster"
        (two obstacles more than BYPASS_CLUSTER_GAP apart used to be
        treated as unrelated events): a roadwork/accident scene can have
        props spread wider than any fixed gap threshold (a cone near the
        start, a warning sign much further along), and chaining silently
        dropped whichever ones didn't fit the chain -- which is what let
        the agent size its manoeuvre for the nearest prop only and clip a
        static.prop.trafficwarning sitting further down the same stretch.
        Obstacles behind the agent are excluded (see `_is_ahead_of_vehicle`).

            :param waypoint: the agent's current waypoint
            :param max_distance: search radius (default BYPASS_DETECTION_DISTANCE)
            :return: list of static obstacles ahead, nearest first
        """
        max_distance = self.BYPASS_DETECTION_DISTANCE if max_distance is None else max_distance
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=max_distance)
        static_list = [o for o in obstacle_list
                       if 'static.prop' in o.type_id and self._is_ahead_of_vehicle(waypoint, o)]
        return sorted(static_list, key=lambda o: o.get_location().distance(waypoint.transform.location))

    def _farthest_obstacle_distance(self, waypoint):
        """
        Distance (m) from the agent to the farthest static obstacle
        currently ahead within range -- sizing a bypass for only the
        closest prop strands the agent among others still further ahead.

            :param waypoint: the agent's current waypoint
            :return: distance (m), or 0 if no obstacle is ahead
        """
        obstacles = self._obstacles_ahead(waypoint)
        if not obstacles:
            return 0.0
        return max(o.get_location().distance(waypoint.transform.location) for o in obstacles)

    def _lane_max_offset(self, waypoint):
        """
        Largest lateral offset (m) the agent can apply while staying
        inside its own lane: half the lane width minus the vehicle's own
        half-width and a safety margin.

            :param waypoint: the agent's current waypoint
            :return: max offset magnitude (m), may be <= 0 on a narrow lane
        """
        vehicle_half_width = self._vehicle.bounding_box.extent.y
        return waypoint.lane_width / 2 - vehicle_half_width - self.BYPASS_OFFSET_MARGIN

    def _obstacle_lateral_offset(self, waypoint, obstacle):
        """
        Signed lateral distance (m) from the lane centerline to the
        obstacle, projected on the lane's right vector (positive = the
        obstacle sits to the right of center).

            :param waypoint: the agent's current waypoint
            :param obstacle: the static prop to clear
            :return: signed lateral offset (m)
        """
        obstacle_loc = obstacle.get_location()
        wp_loc = waypoint.transform.location
        dx, dy = obstacle_loc.x - wp_loc.x, obstacle_loc.y - wp_loc.y
        right = waypoint.transform.get_right_vector()
        return dx * right.x + dy * right.y

    def _obstacle_side_clearance(self, waypoint, obstacle):
        """
        Signed clearance (m, StanleyLateralController convention: positive
        = right) this single obstacle demands to be cleared while staying
        in-lane: how far past its own edge, on whichever side it already
        leans towards, the agent must offset.

            :param waypoint: the agent's current waypoint
            :param obstacle: the static prop to size clearance for
            :return: signed clearance in metres
        """
        lateral = self._obstacle_lateral_offset(waypoint, obstacle)
        half_width = obstacle.bounding_box.extent.y
        needed = abs(lateral) + half_width + self.BYPASS_OFFSET_CLEARANCE
        return needed if lateral >= 0 else -needed

    def _widest_bypass_offset(self, waypoint):
        """
        Offset (m) sized to clear whichever SINGLE obstacle ahead demands
        the most room -- scanned across every static obstacle currently
        within range (see `_obstacles_ahead`), not just the ones close
        enough together to have chained into one "cluster" under the old
        approach. Staying in-lane like this instead of a full lane change
        onto oncoming traffic is what avoids triggering OutsideRouteLanesTest
        for the duration of the manoeuvre. Returns None if the widest
        obstacle takes up too much of the lane for an in-lane nudge to be
        safe, in which case a full opposite-lane bypass is the fallback.

            :param waypoint: the agent's current waypoint
            :return: offset in metres, or None
        """
        obstacles = self._obstacles_ahead(waypoint)
        if not obstacles:
            return None

        max_offset = self._lane_max_offset(waypoint)
        if max_offset <= 0:
            return None

        widest = max(
            (self._obstacle_side_clearance(waypoint, o) for o in obstacles), key=abs)
        if abs(widest) > max_offset:
            return None
        return widest

    def _set_lane_offset(self, offset):
        """
        Applies a lateral offset to the local planner's Stanley controller
        without touching the route plan -- the agent keeps following the
        same lane's waypoints, just displaced sideways, which is why this
        stays inside the route's lane polygon (no OutsideRouteLanesTest hit).

            :param offset: signed lateral offset in metres (0 to cancel)
        """
        self._local_planner._vehicle_controller._lat_controller._offset = offset

    def _bypass_refresh_nudge(self, waypoint):
        """
        Recomputes the in-lane offset every tick while nudging, sized to
        the widest obstacle currently ahead (see `_widest_bypass_offset`),
        instead of freezing a value captured once at detection time --
        static props further down the same stretch can otherwise demand
        more room than what was originally computed.

            :param waypoint: the agent's current waypoint
            :return: True if a valid offset was applied, False if the
                     current obstacle set no longer fits an in-lane nudge
        """
        offset = self._widest_bypass_offset(waypoint)
        if offset is None:
            return False
        self._set_lane_offset(offset)
        return True

    def _forward_on_lane(self, lane_waypoint, same_direction, distance):
        """
        Waypoint `distance` metres ahead of `lane_waypoint`, in the EGO
        VEHICLE's own direction of travel -- not necessarily the lane's
        own driving direction. On a same-direction lane (`same_direction`
        True, e.g. the right lane), the lane's forward direction matches
        ours, so `.next` is correct. On the opposite/oncoming lane used
        as the last-resort bypass side, the lane's own forward direction
        is reversed relative to ours: calling `.next` there returns a
        point BEHIND the ego, which is exactly what made the global route
        planner send the agent into a reversing manoeuvre while
        "bypassing" obstacle #3 (see the tick-by-tick log: x started
        increasing again right after the lane shift). `.previous` on
        that lane is the one that keeps going the way we're already
        headed.

            :param lane_waypoint: waypoint on the target lane
            :param same_direction: True if that lane shares our direction of travel
            :param distance: distance to look ahead (m)
            :return: list of waypoints (possibly empty)
        """
        return lane_waypoint.next(distance) if same_direction else lane_waypoint.previous(distance)

    def _build_lane_shift_path(self, current_waypoint, target_lane_wpt, same_direction, distance):
        """
        Builds an explicit (waypoint, RoadOption) list from the agent's
        current position onto `target_lane_wpt`'s lane, sampled every
        `_sampling_resolution` metres for `distance` metres further along
        that lane IN OUR OWN DIRECTION OF TRAVEL (see `_forward_on_lane`).

        Used for both legs of a full bypass (crossing onto the bypass lane
        and crossing back) INSTEAD of a single `set_destination` end
        location. The reason: `GlobalRoutePlanner.trace_route` picks its
        own intermediate waypoint for a lane-change edge from the target
        lane's topology `path` array (see `global_route_planner.py`,
        `_build_topology`/`trace_route`), which is sampled along that
        lane's OWN forward direction -- reversed relative to ours whenever
        the target is the oncoming lane (the only option on a plain
        two-way road, i.e. every `*TwoWays` scenario). That reversed
        indexing can select a point BEHIND the agent instead of ahead of
        it, which is why the trajectory barely deviated before running
        straight into the obstacle: the "manoeuvre" was, geometrically,
        barely a manoeuvre at all. Building the sequence ourselves with
        `_forward_on_lane` (already fixed to respect our actual direction
        of travel) sidesteps that shared-code ambiguity entirely, and
        gives a smooth multi-waypoint path instead of a single 2-point
        jump for the Stanley controller to track.

            :param current_waypoint: the agent's current waypoint
            :param target_lane_wpt: waypoint on the lane to shift onto
            :param same_direction: True if that lane shares our direction
                of travel (see `_forward_on_lane`)
            :param distance: how far to extend the path along the target
                lane, in our direction of travel (m)
            :return: list of (carla.Waypoint, RoadOption), or None if the
                     target lane doesn't extend far enough to build a path
        """
        path = [(current_waypoint, RoadOption.LANEFOLLOW),
                (target_lane_wpt, RoadOption.LANEFOLLOW)]
        travelled = 0.0
        current = target_lane_wpt
        while travelled < distance:
            step = min(self._sampling_resolution, distance - travelled)
            ahead = self._forward_on_lane(current, same_direction, step)
            if not ahead:
                break
            current = ahead[0]
            path.append((current, RoadOption.LANEFOLLOW))
            travelled += step
        if len(path) < 2:
            return None
        return path

    def _bypass_remaining_distance(self, waypoint):
        """
        Distance still to cover, from the agent's CURRENT position, before
        `_bypass_progress_clear` will consider the obstacle cluster
        cleared -- i.e. the same `_bypass_reach + BYPASS_CLEAR_MARGIN`
        threshold (measured from the manoeuvre's origin), converted into
        "distance from here" so a manually-built path (see
        `_build_lane_shift_path`) is guaranteed at least as long as the
        state machine needs it to be.

        Without this, sizing the path from the obstacle distance measured
        again at manoeuvre-start time (smaller than at detection time,
        since the agent kept approaching while waiting for a gap) could
        produce a path shorter than `_bypass_progress_clear` needs -- the
        local planner's queue would run dry before the state machine
        decides the manoeuvre is done, leaving the agent braked to a
        stop (see `LocalPlanner.run_step`: an empty queue means a full
        brake) instead of continuing back onto the original lane.

            :param waypoint: the agent's current waypoint
            :return: distance in metres, floored at BYPASS_CLEAR_MARGIN
        """
        target = self._bypass_reach + self.BYPASS_CLEAR_MARGIN
        if self._bypass_origin_waypoint is not None:
            travelled = waypoint.transform.location.distance(
                self._bypass_origin_waypoint.transform.location)
            target -= travelled
        return max(target, self.BYPASS_CLEAR_MARGIN)

    def _start_bypass_maneuver(self, waypoint):
        """
        Triggers a lateral shift to the target lane (see `_bypass_side`)
        to bypass the obstacle cluster, using a manually-built path (see
        `_build_lane_shift_path`) rather than `set_destination` -- see
        that method's docstring for why.

            :param waypoint: the agent's current waypoint
            :return: True if the manoeuvre was actually started
        """
        same_direction = self._bypass_side(waypoint) == 1
        target_wpt = waypoint.get_right_lane() if same_direction else waypoint.get_left_lane()
        if target_wpt is None:
            return False
        distance = self._bypass_remaining_distance(waypoint)
        path = self._build_lane_shift_path(waypoint, target_wpt, same_direction, distance)
        if path is None:
            return False
        self._local_planner.set_global_plan(path, stop_waypoint_creation=True, clean_queue=True)
        return True

    def _resume_original_lane(self, waypoint):
        """
        Issues a manually-built path (see `_build_lane_shift_path`) back
        onto the original lane, well past the cluster -- same rationale
        as `_start_bypass_maneuver`, applied to the return leg (crossing
        back has the exact same direction-ambiguity risk as crossing out,
        since `_forward_on_lane`/`.previous` vs `.next` depends only on
        which lane is being entered, not which leg of the manoeuvre this
        is).

        The target lane point handed to `_build_lane_shift_path` is NOT
        `_bypass_origin_waypoint` itself: by the time the manoeuvre is
        done, the agent has driven well past it, so using it directly
        would build a path whose first steps point BACKWARDS (origin
        sitting behind the agent's current position). Advancing from the
        origin by the distance already travelled since it was captured
        first locates a point abeam of the agent's current position on
        the original lane -- from there, `BYPASS_RESUME_DISTANCE` extends
        forward as intended.

            :param waypoint: the agent's current waypoint
            :return: True if a new path was set
        """
        origin = self._bypass_origin_waypoint or waypoint
        travelled = (waypoint.transform.location.distance(origin.transform.location)
                     if self._bypass_origin_waypoint is not None else 0.0)
        abeam = origin.next(travelled)
        target_lane_wpt = abeam[0] if abeam else origin
        path = self._build_lane_shift_path(
            waypoint, target_lane_wpt, same_direction=True, distance=self.BYPASS_RESUME_DISTANCE)
        if path is None:
            return False
        self._local_planner.set_global_plan(path, stop_waypoint_creation=True, clean_queue=True)
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

    def _bypass_progress_clear(self, waypoint):
        """
        Distance-based replacement for `_obstacle_cleared`, used once the
        agent has actually moved onto the opposite lane ('overtaking' /
        'returning'). `_blocking_obstacle_ahead` filters candidates by the
        agent's CURRENT lane, so as soon as the agent steers left it wrongly
        reports the whole cluster as gone -- well before it is physically
        behind the agent. Comparing distance travelled since detection
        against the farthest tracked obstacle's distance (captured once,
        see `_farthest_obstacle_distance`)
        is immune to that lane-relative blind spot.

            :param waypoint: the agent's current waypoint
            :return: True once far enough past the origin to clear the cluster
        """
        if self._bypass_origin_waypoint is None:
            return True
        travelled = waypoint.transform.location.distance(
            self._bypass_origin_waypoint.transform.location)
        return travelled >= self._bypass_reach + self.BYPASS_CLEAR_MARGIN

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

    def _nudge_convergence_speed(self, waypoint, base_speed):
        """
        Speed cap for an in-lane nudge, scaled down further the larger the
        required offset is relative to what the lane allows (see
        `_bypass_convergence_speed` for why a lower speed gives the Stanley
        controller more correction authority, not less).

            :param waypoint: the agent's current waypoint
            :param base_speed: uncapped manoeuvre speed (see `_bypass_target_speed`)
            :return: target speed in km/h
        """
        offset = self._widest_bypass_offset(waypoint)
        max_offset = self._lane_max_offset(waypoint)
        if offset is None or max_offset <= 0:
            return base_speed

        severity = min(abs(offset) / max_offset, 1.0)
        return base_speed - severity * (base_speed - self.BYPASS_MIN_MANEUVER_SPEED)

    def _overtaking_transition_speed(self, waypoint, base_speed):
        """
        Speed cap for the first `BYPASS_TRANSITION_DISTANCE` metres of a
        full lane-level bypass, i.e. the unsmoothed diagonal produced by
        the global route planner's lane-change edge (see
        `BYPASS_TRANSITION_DISTANCE`). Once far enough past the origin,
        the agent is assumed to have straightened out onto the target
        lane and can resume the normal manoeuvre speed.

            :param waypoint: the agent's current waypoint
            :param base_speed: uncapped manoeuvre speed (see `_bypass_target_speed`)
            :return: target speed in km/h
        """
        if self._bypass_origin_waypoint is None:
            return base_speed
        travelled = waypoint.transform.location.distance(
            self._bypass_origin_waypoint.transform.location)
        if travelled < self.BYPASS_TRANSITION_DISTANCE:
            return self.BYPASS_MIN_MANEUVER_SPEED
        return base_speed

    def _waiting_gap_speed(self, waypoint, base_speed):
        """
        Speed while 'waiting_gap' -- a blocking obstacle is confirmed but
        no safe gap in oncoming traffic has been found yet to start the
        full lane-level bypass. Per the strategy doc's guidance for
        ConstructionObstacleTwoWays/ParkedObstacleTwoWays/AccidentTwoWays
        ("ralentir et attendre une fenetre de degagement avant de
        contourner"): ramps down linearly from `base_speed` to a full
        stop as the tracked obstacle's distance shrinks from
        `BYPASS_DETECTION_DISTANCE` down to `BYPASS_WAITING_STOP_DISTANCE`.

        Without this, the agent cruised at `base_speed` for the entire
        wait -- since nothing capped speed in this state before -- and
        only braked once `_bypass_forward_clear`'s BYPASS_FORWARD_MARGIN
        (8 m) threshold tripped, far too late to stop from ~25 km/h (see
        the log: no "Gap found" message ever appears before the
        collision, meaning the agent was still 'waiting_gap' -- still
        cruising -- right up to impact).

            :param waypoint: the agent's current waypoint
            :param base_speed: uncapped manoeuvre speed (see `_bypass_target_speed`)
            :return: target speed in km/h
        """
        _, _, distance = self._blocking_obstacle_ahead(waypoint)
        if distance is None or distance < 0:
            return base_speed

        stop_distance = self.BYPASS_WAITING_STOP_DISTANCE
        if distance <= stop_distance:
            return 0.0

        detection_range = self.BYPASS_DETECTION_DISTANCE
        if distance >= detection_range or detection_range <= stop_distance:
            return base_speed

        severity = (detection_range - distance) / (detection_range - stop_distance)
        return base_speed * (1.0 - severity)

    def _bypass_convergence_speed(self, waypoint):
        """
        Speed cap applied while a bypass manoeuvre needs extra lateral
        correction authority from the Stanley controller, or while it
        must decelerate safely towards an obstacle it hasn't started
        going around yet.

        The Stanley lateral controller's correction term is
        `atan(K_V * lateral_error / (K_S + speed))`: for the same
        crosstrack error, a LOWER speed gives a STRONGER correction, not
        a weaker one. `_bypass_target_speed` alone caps every manoeuvre at
        the same speed regardless of how much lateral distance it needs
        to cover, which is what let the agent still be mid-drift, short
        of the width it needed, by the time it reached the obstacle --
        for an in-lane nudge (`_nudge_convergence_speed`) as much as for
        the sharp diagonal at the start of a full lane change
        (`_overtaking_transition_speed`). Separately, while waiting for a
        gap (`_waiting_gap_speed`) the agent must decelerate towards the
        obstacle it hasn't started bypassing yet, rather than cruising
        at full manoeuvre speed until a last-resort emergency brake.

            :param waypoint: the agent's current waypoint
            :return: target speed in km/h
        """
        base_speed = self._bypass_target_speed()

        if self._bypass_state == 'nudging':
            return self._nudge_convergence_speed(waypoint, base_speed)

        if self._bypass_state == 'overtaking':
            return self._overtaking_transition_speed(waypoint, base_speed)

        if self._bypass_state == 'waiting_gap':
            return self._waiting_gap_speed(waypoint, base_speed)

        return base_speed

    def _bypass_forward_clear(self, waypoint):
        """
        Checks for a vehicle immediately ahead in the lane currently used
        during the manoeuvre (another vehicle merging into it, for
        instance). While actively going around the obstacle ('overtaking'/
        'returning'/'nudging'), only the specific obstacle(s) captured in
        `_bypass_cluster_ids` at detection time (plus a confirmed stalled
        vehicle) are excluded -- they are already handled by the bypass
        state machine itself (`_obstacle_cleared`), and re-checking them
        here caused the agent to brake to a stop right against the very
        obstacle it was going around. This used to exclude every
        static.prop by TYPE instead of by id: whenever the tracked cluster
        legitimately spanned a long stretch,
        that blanket exclusion made the agent blind to any *other* static
        prop appearing anywhere in that stretch too -- which is what let
        it drive straight into a static.prop.trafficwarning it was never
        actually bypassing. Before detection ('idle'/'waiting_gap'), no
        exclusion applies at all, so the agent still slows down and stops
        approaching the obstacle while it waits for a safe gap.

            :param waypoint: the agent's current waypoint
            :return: True if it is safe to keep driving, False if it must brake
        """
        vehicle_list = self._build_obstacle_list(waypoint, max_distance=self.OBSTACLE_MAX_DISTANCE)
        if self._bypass_state in ('overtaking', 'returning', 'nudging'):
            ignored_ids = self._bypass_cluster_ids | {self._stalled_vehicle_id}
            vehicle_list = [v for v in vehicle_list if v.id not in ignored_ids]
        vehicle_state, vehicle, distance = self._forward_obstacle_detected(vehicle_list)
        if not vehicle_state:
            return True
        margin = max(vehicle.bounding_box.extent.x, vehicle.bounding_box.extent.y)
        return (distance - margin) >= self.BYPASS_FORWARD_MARGIN

    def _fallback_to_full_bypass(self, waypoint):
        """
        Abandons an in-lane nudge in favour of a full lane-level bypass.

        Clears the in-lane offset applied while nudging -- left in place,
        it lingers as a stale lateral bias once `_start_bypass_maneuver`
        sets a destination on the opposite lane: the new trajectory
        already accounts for the full lane shift, so adding the old
        nudge offset on top of it either widens or narrows the actual
        clearance unpredictably. This is what let the agent still clip a
        static.prop.trafficwarning that needed more room than an in-lane
        nudge could give: `_bypass_refresh_nudge` correctly detected the
        obstacle no longer fit an in-lane offset, but the offset itself
        was never cancelled before switching to the full bypass path.

            :param waypoint: the agent's current waypoint
        """
        self._set_lane_offset(0.0)
        self._bypass_state = 'waiting_gap'
        self._log_bypass_transition('detected', waypoint)

    def _reset_bypass_state(self):
        """Resets the status of the bypass module."""
        self._bypass_state = 'idle'
        self._bypass_origin_waypoint = None
        self._bypass_tick_counter = 0
        self._stalled_vehicle_id = None
        self._stalled_tick_counter = 0
        self._bypass_reach = 0.0
        self._bypass_cluster_ids = frozenset()
        self._set_lane_offset(0.0)
        self._exit_bypass_caution()

    def bypass_diagnostics(self, waypoint):
        """
        Snapshot of every measurement the bypass manoeuvre depends on, at
        the current tick: lane/vehicle geometry, the static obstacle
        cluster ahead, and the oncoming-traffic situation on the opposite
        lane. Pure read -- it recomputes nothing the bypass logic doesn't
        already compute elsewhere (`_obstacles_ahead`, `_oncoming_lane_
        obstacle`, `_gap_is_safe`, etc.) and changes no state. Meant to
        be printed/logged (see `log_bypass_diagnostics`) when deciding
        what the manoeuvre should do next, without needing a fresh
        simulation run to see the raw numbers behind a given decision.

            :param waypoint: the agent's current waypoint
            :return: dict with the following keys:
                - 'bypass_state': current state of the bypass state machine
                - 'lane_width': width (m) of the agent's current lane
                - 'vehicle_width' / 'vehicle_half_width': ego vehicle's
                  own width (m), from its bounding box
                - 'max_in_lane_offset': largest lateral offset (m) an
                  in-lane nudge could use without leaving the lane (see
                  `_lane_max_offset`); <= 0 means the lane is too narrow
                  for any in-lane nudge
                - 'static_obstacles_ahead': list of dicts, nearest first,
                  one per tracked static obstacle, each with 'id',
                  'type_id', 'distance', 'lateral_offset' (signed, +right)
                  and 'side_clearance_needed' (signed offset an in-lane
                  nudge would need to clear THIS obstacle alone)
                - 'farthest_obstacle_distance': distance (m) to the
                  farthest tracked obstacle -- what a full bypass must
                  clear before merging back
                - 'widest_in_lane_offset_needed': the offset an in-lane
                  nudge would need to clear every tracked obstacle at
                  once, or None if none fits (see `_widest_bypass_offset`)
                - 'right_lane': {'usable': bool, 'clear': bool or None}
                - 'oncoming': {'detected': bool, 'vehicle_id', 'distance',
                  'speed_kmh', 'time_to_arrival_s', 'gap_safe'} -- the
                  nearest vehicle on the opposite lane used for a
                  left-side bypass, and whether `_gap_is_safe` currently
                  considers that gap safe to cross into
        """
        vehicle_half_width = self._vehicle.bounding_box.extent.y

        static_obstacles = []
        for obstacle in self._obstacles_ahead(waypoint):
            static_obstacles.append({
                'id': obstacle.id,
                'type_id': obstacle.type_id,
                'distance': obstacle.get_location().distance(waypoint.transform.location),
                'lateral_offset': self._obstacle_lateral_offset(waypoint, obstacle),
                'half_width': obstacle.bounding_box.extent.y,
                'side_clearance_needed': self._obstacle_side_clearance(waypoint, obstacle),
            })

        oncoming_state, oncoming_vehicle, oncoming_distance = self._oncoming_lane_obstacle(waypoint)
        oncoming_speed = get_speed(oncoming_vehicle) if oncoming_vehicle is not None else None
        oncoming_ttc = None
        if oncoming_state and oncoming_speed and oncoming_speed > 0:
            oncoming_ttc = oncoming_distance / (oncoming_speed / 3.6)

        right_wpt = self._right_lane_usable(waypoint)

        return {
            'bypass_state': self._bypass_state,
            'lane_width': waypoint.lane_width,
            'vehicle_width': vehicle_half_width * 2,
            'vehicle_half_width': vehicle_half_width,
            'max_in_lane_offset': self._lane_max_offset(waypoint),
            'static_obstacles_ahead': static_obstacles,
            'farthest_obstacle_distance': self._farthest_obstacle_distance(waypoint),
            'widest_in_lane_offset_needed': self._widest_bypass_offset(waypoint),
            'right_lane': {
                'usable': right_wpt is not None,
                'clear': self._right_lane_clear(waypoint) if right_wpt is not None else None,
            },
            'oncoming': {
                'detected': oncoming_state,
                'vehicle_id': oncoming_vehicle.id if oncoming_vehicle is not None else None,
                'distance': oncoming_distance if oncoming_state else None,
                'speed_kmh': oncoming_speed,
                'time_to_arrival_s': oncoming_ttc,
                'gap_safe': self._gap_is_safe(
                    oncoming_distance if oncoming_state else -1,
                    oncoming_speed if oncoming_speed is not None else 0),
            },
        }

    def format_bypass_diagnostics(self, report):
        """
        Renders a `bypass_diagnostics` report as a compact, greppable,
        multi-line string (mirrors the `[BYPASS] ...` log style already
        used by `_log_bypass_transition`).

            :param report: dict returned by `bypass_diagnostics`
            :return: formatted string, ready to `print`
        """
        lines = [
            f"[BYPASS-INFO] state={report['bypass_state']} "
            f"lane_width={report['lane_width']:.2f}m "
            f"vehicle_width={report['vehicle_width']:.2f}m "
            f"max_in_lane_offset={report['max_in_lane_offset']:.2f}m",
        ]

        if report['static_obstacles_ahead']:
            for obs in report['static_obstacles_ahead']:
                lines.append(
                    f"[BYPASS-INFO]   obstacle id={obs['id']} type={obs['type_id']} "
                    f"distance={obs['distance']:.1f}m lateral_offset={obs['lateral_offset']:.2f}m "
                    f"half_width={obs['half_width']:.2f}m "
                    f"side_clearance_needed={obs['side_clearance_needed']:.2f}m")
        else:
            lines.append("[BYPASS-INFO]   no static obstacle currently tracked")

        lines.append(
            f"[BYPASS-INFO]   farthest_obstacle_distance={report['farthest_obstacle_distance']:.1f}m "
            f"widest_in_lane_offset_needed="
            f"{report['widest_in_lane_offset_needed']}")

        right = report['right_lane']
        lines.append(f"[BYPASS-INFO]   right_lane usable={right['usable']} clear={right['clear']}")

        oncoming = report['oncoming']
        if oncoming['detected']:
            ttc = oncoming['time_to_arrival_s']
            ttc_str = f"{ttc:.1f}s" if ttc is not None else "n/a"
            lines.append(
                f"[BYPASS-INFO]   oncoming vehicle_id={oncoming['vehicle_id']} "
                f"distance={oncoming['distance']:.1f}m speed={oncoming['speed_kmh']:.1f}km/h "
                f"time_to_arrival={ttc_str} gap_safe={oncoming['gap_safe']}")
        else:
            lines.append(f"[BYPASS-INFO]   oncoming lane clear (gap_safe={oncoming['gap_safe']})")

        return "\n".join(lines)

    def log_bypass_diagnostics(self, waypoint):
        """
        Convenience wrapper: computes `bypass_diagnostics` and prints it
        via `format_bypass_diagnostics`. Not called automatically from
        `run_step` -- toggle `BYPASS_DEBUG = True` on the class (or an
        instance) if you want it printed every tick a manoeuvre is active,
        without needing to add print statements by hand each time you
        want to see the numbers behind a decision.

            :param waypoint: the agent's current waypoint
            :return: the report dict (same as `bypass_diagnostics`),
                     already printed
        """
        report = self.bypass_diagnostics(waypoint)
        print(self.format_bypass_diagnostics(report))
        return report

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
                self._bypass_reach = self._farthest_obstacle_distance(waypoint)
                self._bypass_cluster_ids = frozenset(
                    o.id for o in self._obstacles_ahead(waypoint))
                self._enter_bypass_caution()
                offset = self._widest_bypass_offset(waypoint)
                if offset is not None:
                    self._set_lane_offset(offset)
                    self._bypass_state = 'nudging'
                    self._log_bypass_transition('nudging', waypoint)
                else:
                    self._bypass_state = 'waiting_gap'
                    self._log_bypass_transition('detected', waypoint)
            return self._bypass_state != 'idle'

        if self._bypass_timed_out():
            self._log_bypass_transition('timeout', waypoint)
            self._reset_bypass_state()
            return False

        if self._bypass_state == 'nudging':
            if self._obstacle_cleared(waypoint):
                self._log_bypass_transition('success', waypoint)
                self._reset_bypass_state()
                return False
            if not self._bypass_refresh_nudge(waypoint):
                # The obstacle set ahead no longer fits an in-lane nudge
                # (e.g. a new prop further down the scene needs more
                # clearance than the lane allows) -- fall back to a full
                # lane-level bypass.
                self._fallback_to_full_bypass(waypoint)
            return True

        if self._bypass_state == 'waiting_gap':
            if self._can_start_bypass(waypoint):
                if not self._start_bypass_maneuver(waypoint):
                    # No opposite lane available at this point (edge case
                    # that used to crash the run) -- abort instead of
                    # retrying against a target that will never exist.
                    self._log_bypass_transition('aborted', waypoint)
                    self._reset_bypass_state()
                    return False
                self._bypass_state = 'overtaking'
                self._log_bypass_transition('gap_found', waypoint)
            return True

        if self._bypass_state == 'overtaking':
            if self._bypass_progress_clear(waypoint):
                if not self._resume_original_lane(waypoint):
                    # Same edge case as above, at the other end of the
                    # manoeuvre: nowhere to resume to -- abort rather than
                    # leave the planner to run dry and freeze in place.
                    self._log_bypass_transition('aborted', waypoint)
                    self._reset_bypass_state()
                    return False
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
        self._local_planner.set_speed(self._bypass_convergence_speed(waypoint))
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
        if self.BYPASS_DEBUG and self._bypass_state != 'idle':
            self.log_bypass_diagnostics(ego_vehicle_wp)
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