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

    JUNCTION_STALL_EXCLUSION_DISTANCE = 25.0   

    BYPASS_DETECTION_DISTANCE = 80
    BYPASS_MIN_GAP_TIME = 4.0
    BYPASS_LANE_OVERLAP_MARGIN = 0.3   
    BYPASS_OVERTAKE_MARGIN = 8.0  

    BYPASS_ABORT_RESUME_SPEED_THRESHOLD = 8.0   
    BYPASS_ABORT_RESUME_TICKS = 40   

    STATE_LOG_INTERVAL = 20
    BYPASS_TIMEOUT_TICKS = 800

    JUNCTION_DETECTION_DISTANCE = 30
    JUNCTION_MIN_GAP_TIME = 3.0
    JUNCTION_TIMEOUT_TICKS = 300

    STALL_SPEED_THRESHOLD = 1.0    
    STALL_TIMEOUT_TICKS = 150      
    EGO_BLOCKED_SPEED_THRESHOLD = 2.0    
    EGO_BLOCKED_CONFIRM_TICKS = 100 
    BYPASS_MAX_CONSECUTIVE_RETRIES = 2
    BYPASS_GIVEUP_COOLDOWN_TICKS = 300   

    STALL_MIN_EGO_MOVED_SPEED_KMH = 10.0   
    STALL_STARTUP_HARD_BLOCK_TICKS = 400   
    STALL_STARTUP_GRACE_TICKS = 200   

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
        self._bypass_start_tick = None
        self._last_bypass_transition_tick = None   # see `_in_bypass_maneuver_grace_period`
        self._bypass_target_actor = None       
        self._bypass_target_resumed_tick_counter = 0   # see `_update_bypass_target_resumed_tracking`

        self._actors = None

        self._tick_count = 0
        self._scenario_result = None

        # Cycle : 'idle' -> 'waiting_clear' -> 'crossing' -> 'idle'
        self._junction_state = 'idle'
        self._junction_start_tick = None

        # Stalled-vehicle tracking (see STALL_TIMEOUT_TICKS)
        self._stalled_vehicle_id = None
        self._stalled_tick_counter = 0
        self._ego_blocked_tick_counter = 0
        self._ego_has_moved_normally = False   # see STALL_MIN_EGO_MOVED_SPEED_KMH  

        self._bypass_giveup_id = None
        self._bypass_giveup_tick = None
        self._last_bypass_failed_id = None
        self._bypass_retry_count = 0

        # Pedestrian wait tracking (see PEDESTRIAN_WAIT_TIMEOUT_TICKS)
        self._pedestrian_wait_id = None
        self._pedestrian_wait_tick_counter = 0

        # Cycle : 'idle' -> 'stabilizing' -> 'idle' (Block 5: ControlLoss)
        self._control_loss_state = 'idle'
        self._control_loss_start_tick = None   # absolute self._tick_count when stabilizing began
        self._control_loss_pending_ticks = 0   # consecutive detections while still 'idle' (debounce)

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

    def _log_bypass_transition(self, event, waypoint=None, obstacle=None):
        """
        A single entry point for all messages relating to the obstacle-avoidance cycle, 
        to ensure a consistent format and to make it easy to grep the logs 
        (e.g. test success/failure).

            :param event: 'detected' | 'gap_found' | 'success' | 'timeout' | 'no_lane'
            :param waypoint: the agent’s current waypoint (optional)
            :param obstacle: the obstacle actor that triggered 'detected' (optional) - logging its
                type_id makes it possible to tell a real scenario-spawned obstacle apart from
                unrelated map dressing without attaching a debugger (see `_is_obstacle_in_lane`).
        """
        loc = waypoint.transform.location if waypoint is not None else None
        pos = f" pos=({loc.x:.1f}, {loc.y:.1f})" if loc is not None else ""
        obstacle_info = f" obstacle={obstacle.type_id}(id={obstacle.id})" if obstacle is not None else ""
        messages = {
            'detected': f"[BYPASS] Obstacle detected{pos}{obstacle_info} -> searching for a gap",
            'gap_found': f"[BYPASS] Gap found{pos} -> start of manoeuvre",
            'success': f"[BYPASS] TEST SUCCESSFUL: obstacle bypassed, back on track{pos}",
            'timeout': f"[BYPASS] TEST FAILED: manoeuvre timed out{pos}"
                       f"(stuck in '{self._bypass_state}')",
            'no_lane': f"[BYPASS] TEST FAILED: no usable opposite lane{pos} -> abandoning bypass",
            'resumed_normal': f"[BYPASS] Target resumed normal driving{pos}{obstacle_info} "
                               f"-> was never really stalled, aborting manoeuvre and reverting "
                               f"to car-following",
            'cyclist_gone': f"[BYPASS] Tracked cyclist no longer detected{pos}{obstacle_info} "
                             f"-> aborting manoeuvre and reverting to normal behaviour",
        }
        print(messages[event])
        if event == 'success':
            self._scenario_result = True
        elif event in ('timeout', 'no_lane'):
            self._scenario_result = False
        # 'resumed_normal' and 'cyclist_gone' are correct self-diagnoses, not scenario
        # failures - they deliberately leave self._scenario_result untouched.

    def _bypass_timed_out(self):
        """
        Indicates whether the bypass manoeuvre has been running for too
        long (enables detection of a scenario failure without waiting for
        the route's global timeout).

        Measured against `self._tick_count` (incremented unconditionally at
        the very start of every `run_step`) rather than a counter
        incremented only while this method is actually called - see the
        identical fix and rationale on `_control_loss_timed_out`: a
        per-call counter can freeze for an arbitrarily long real-world time
        whenever a higher-priority branch earlier in `run_step` (pedestrian
        wait, control-loss stabilizing...) keeps returning before ever
        reaching this check.

            :return: True if the operation is considered to be blocked
        """
        return (self._tick_count - self._bypass_start_tick) > self.BYPASS_TIMEOUT_TICKS

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

    def _dynamic_forward_distance(self, base_distance):
        """
        Widens a static base distance by how far the vehicle travels in
        DETECTION_SPEED_MARGIN_SECONDS at its CURRENT speed, so cruising
        faster automatically buys more reaction room instead of a range
        tied only to the road's speed limit (§3.2/§4.1).

            :param base_distance: static minimum distance to widen
            :return: base_distance plus a speed-proportional margin (m)
        """
        speed_ms = self._speed / 3.6
        return base_distance + speed_ms * self.DETECTION_SPEED_MARGIN_SECONDS

    def _effective_braking_distance(self):
        """
        The behavior's base `braking_distance`, extended by the vehicle's
        current speed so the emergency-stop trigger keeps a comparable time
        margin at higher speed rather than a fixed number of meters
        regardless of how fast the agent is going.

            :return: effective emergency-stop trigger distance (m)
        """
        speed_ms = self._speed / 3.6
        return self._behavior.braking_distance + speed_ms * self.BRAKING_SPEED_MARGIN_SECONDS

    def _collision_detection_range(self):
        """
        Forward range used both to gather obstacle candidates
        (`_build_obstacle_list`) and to test them (`_forward_obstacle_detected`):
        kept identical so widening one never leaves the other as a hidden
        bottleneck (a fixed 45 m candidate-gathering radius would silently
        cap the detection range no matter how far `_dynamic_forward_distance`
        pushes it out).

            :return: forward detection range (m)
        """
        base = max(self.OBSTACLE_MAX_DISTANCE,
                   self._behavior.min_proximity_threshold,
                   self._speed_limit / 3)
        return self._dynamic_forward_distance(base)

    def _road_projection(self, actor, waypoint):
        """
        Projects `actor` onto the road's own local frame at `waypoint`:
        longitudinal distance ahead along the road heading, and signed
        lateral distance from its centerline. Computed directly from the
        road's heading vector rather than from lane_id, so it works even
        for an actor whose current waypoint isn't (yet) snapped onto the
        driving lane - see `LATERAL_HAZARD_HALF_WIDTH`.

            :param actor: the candidate obstacle
            :param waypoint: the agent's current waypoint
            :return: (longitudinal, lateral) in meters
        """
        yaw = math.radians(waypoint.transform.rotation.yaw)
        forward = (math.cos(yaw), math.sin(yaw))
        right = (-math.sin(yaw), math.cos(yaw))
        loc = actor.get_location()
        origin = waypoint.transform.location
        dx, dy = loc.x - origin.x, loc.y - origin.y
        longitudinal = dx * forward[0] + dy * forward[1]
        lateral = dx * right[0] + dy * right[1]
        return longitudinal, lateral

    def _closest_forward_lateral(self, obstacle_list, waypoint):
        """
        Among `obstacle_list`, finds the closest actor that is ahead of the
        agent (longitudinal > 0) and within `LATERAL_HAZARD_HALF_WIDTH` of
        the lane centerline - factored out of `_lateral_hazard_ahead` so
        the same geometric scan can be restricted to a subset of actors
        (e.g. cyclists only, see `_cyclist_ahead`).

            :param obstacle_list: candidate actors to scan
            :param waypoint: the agent's current waypoint
            :return: tuple (actor, longitudinal_distance), (None, None) if none qualify
        """
        closest = None
        closest_distance = None
        for actor in obstacle_list:
            longitudinal, lateral = self._road_projection(actor, waypoint)
            if longitudinal <= 0:
                continue  # behind the agent
            if abs(lateral) > self.LATERAL_HAZARD_HALF_WIDTH:
                continue  # too far to the side to be a real hazard
            if closest_distance is None or longitudinal < closest_distance:
                closest, closest_distance = actor, longitudinal
        return closest, closest_distance

    def _lateral_hazard_ahead(self, waypoint):
        """
        Direct geometric scan for something approaching the road from the
        side (a cyclist still on the shoulder, a crossing object) within
        the same dynamic range as the forward scan, independent of
        lane_id matching (see the class-level note on
        `LATERAL_HAZARD_HALF_WIDTH`).

            :param waypoint: the agent's current waypoint
            :return: tuple (hazard_state, actor, longitudinal_distance)
        """
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self._collision_detection_range())
        closest, closest_distance = self._closest_forward_lateral(obstacle_list, waypoint)
        if closest is None:
            return False, None, -1
        return True, closest, closest_distance

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
        when cornering (see `_forward_detection_angle`) and a detection range
        that grows with the vehicle's current speed (see `_collision_detection_range`).

            :param vehicle_list: list of obstacles to consider
            :return: tuple (vehicle_state, vehicle, distance)
        """
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

        # Candidate list gathered over the SAME range used to test them
        # (see `_collision_detection_range`), so a static 45 m radius never
        # silently caps the speed-widened forward detection distance below.
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

    def _bbox_adjusted_distance(self, actor, distance):
        """
        Converts a center-to-center distance into an edge-to-edge one using
        both actors' bounding boxes. Factored out of `run_step` so the same
        correction (previously only applied to the plain car-following
        path) is applied consistently everywhere a moving-obstacle distance
        is compared against `_effective_braking_distance()`.

            :param actor: the detected obstacle
            :param distance: raw center-to-center distance (m)
            :return: distance adjusted for both bounding boxes (m)
        """
        return distance - max(
            actor.bounding_box.extent.y, actor.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

    def _react_to_moving_obstacle(self, waypoint, debug=False, ignore_actor_id=None):
        """
        Runs the moving-obstacle check (`collision_and_car_avoid_manager`)
        and, if something is within braking range, returns the control to
        apply (emergency stop or car-following) right away.

            :param waypoint: the agent's current waypoint
            :param debug: boolean for debugging
            :param ignore_actor_id: id of an actor to disregard even if
                detected - used while bypassing so the agent doesn't brake
                for the very obstacle it is deliberately steering around
                (that obstacle stays "ahead" in the forward-detection cone
                for a while even during a correctly-executing manoeuvre;
                treating it as a new hazard would stall the vehicle right
                next to it instead of actually passing it)
            :return: a carla.VehicleControl if a moving obstacle requires
                     immediate action, otherwise None (caller is free to
                     apply its own speed/route for this tick)
        """
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(waypoint)
        if not vehicle_state:
            return None
        if ignore_actor_id is not None and vehicle.id == ignore_actor_id:
            return None

        distance = self._bbox_adjusted_distance(vehicle, distance)
        if distance < self._effective_braking_distance():
            return self.emergency_stop()
        return self.car_following_manager(vehicle, distance, debug=debug)

#----------------------------------------------------------------------------------------------#

    def _is_obstacle_in_lane(self, obstacle, waypoint):
        """
        True only if `obstacle`'s own footprint genuinely overlaps the
        current driving lane - not just its center matching a nearby
        waypoint's lane_id/road_id.

            :param obstacle: the candidate obstacle actor
            :param waypoint: the agent's current waypoint
            :return: True if the obstacle's footprint reaches into the
                     driving lane
        """
        _, lateral = self._road_projection(obstacle, waypoint)
        obstacle_half_width = max(obstacle.bounding_box.extent.x, obstacle.bounding_box.extent.y)
        return (abs(lateral) - obstacle_half_width) < (waypoint.lane_width / 2.0 + self.BYPASS_LANE_OVERLAP_MARGIN)

    def _is_cyclist(self, actor):
        """
        A cyclist is a `vehicle`-type actor in CARLA (not a
        `walker.pedestrian` one), so it has to be told apart from cars by
        its blueprint id (see `CYCLIST_TYPE_KEYWORDS`).

            :param actor: the candidate actor
            :return: True if its blueprint is one of the known bike models
        """
        return any(keyword in actor.type_id for keyword in self.CYCLIST_TYPE_KEYWORDS)

    def _cyclist_ahead(self, waypoint):
        """
        Detects a cyclist directly ahead OR still to the side (shoulder /
        bike lane) of the agent - a cyclist is the one moving-vehicle case
        that must trigger an overtake decision on its own rather than plain
        car-following (user request). Restricted to bike-blueprint actors
        only, so an ordinary car ahead never takes this path.

            :param waypoint: the agent's current waypoint
            :return: tuple (cyclist_state, cyclist, distance)
        """
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        cyclist_list = [a for a in obstacle_list if self._is_cyclist(a)]
        if not cyclist_list:
            return False, None, -1

        forward_state, forward_actor, forward_distance = self._vehicle_obstacle_detected(
            cyclist_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=self._forward_detection_angle())
        lateral_actor, lateral_distance = self._closest_forward_lateral(cyclist_list, waypoint)

        if forward_state and lateral_actor is not None:
            if forward_distance <= lateral_distance:
                return True, forward_actor, forward_distance
            return True, lateral_actor, lateral_distance
        if forward_state:
            return True, forward_actor, forward_distance
        if lateral_actor is not None:
            return True, lateral_actor, lateral_distance
        return False, None, -1

    def _bypass_target_still_present(self, waypoint):
        """
        For a bypass currently targeting a cyclist: True unless that
        specific cyclist has vanished from the scene entirely (destroyed,
        or moved out of detection range) - user request: "revenir à l'état
        normal quand le vélo n'est plus détecté devant ou sur le côté".
        A no-op (always True) for a static obstacle or a confirmed-stalled
        vehicle, neither of which is expected to move away on its own
        (a stalled vehicle driving off is already handled separately, see
        `_bypass_target_resumed_normal_driving`).

            :param waypoint: the agent's current waypoint
            :return: False only if a tracked cyclist has disappeared
        """
        actor = self._bypass_target_actor
        if actor is None or not self._is_cyclist(actor):
            return True
        if not actor.is_alive:
            return False
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        return any(a.id == actor.id for a in obstacle_list)

    def _bypass_target_is_stalled_vehicle(self):
        """
        True only if the memorized bypass target is a genuine stalled
        *vehicle* (AccidentTwoWays / broken-down car), as opposed to a
        static prop (ConstructionObstacleTwoWays / ParkedObstacleTwoWays
        blocker) or a cyclist (HazardAtSideLaneTwoWays). This distinction
        matters because only this category is expected to stay motionless:
        a static prop can never drive off, and a cyclist is deliberately
        overtaken *while* it moves slowly - so any speed-based "it started
        driving again" abort must apply to this category ONLY, never to the
        other two.

            :return: True if the current target is a stalled-type vehicle
        """
        actor = self._bypass_target_actor
        if actor is None:
            return False
        if "static.prop" in actor.type_id:
            return False
        if self._is_cyclist(actor):
            return False
        return True

    def _bypass_target_resumed_before_start(self):
        """
        Instantaneous re-validation used in the 'waiting_gap' phase, right
        before committing to the lane change: True if the tracked target is
        a stalled-type vehicle (see `_bypass_target_is_stalled_vehicle`)
        that is now rolling again above `STALL_SPEED_THRESHOLD`.

        The bypass was justified by that vehicle being *confirmed stalled*;
        if it has since started moving, that premise is gone and the agent
        must NOT pull out into the opposite lane. Otherwise an ordinary lead
        car that merely paused for a few seconds (a queue, dense traffic, a
        scenario-induced hard brake...) gets overtaken the very instant it
        pulls away - exactly the "incoherent overtake" symptom. This is the
        cheapest possible abort: it happens before any lateral move, so
        reverting to plain car-following costs nothing.

        Complements `_bypass_target_resumed_normal_driving`, which only
        kicks in *after* the lane change has begun (the 'overtaking' phase).

            :return: True if the manoeuvre should be aborted before it starts
        """
        if not self._bypass_target_is_stalled_vehicle():
            return False
        if not self._bypass_target_actor.is_alive:
            return False
        return get_speed(self._bypass_target_actor) > self.STALL_SPEED_THRESHOLD

    def _static_obstacle_ahead(self, waypoint):
        """
        Detects a static obstacle (roadworks, accident, parked vehicle)
        blocking the agent’s path, within the detour range.

            :param waypoint: the agent’s current waypoint
            :return: tuple (obstacle_state, obstacle, distance)
        """
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        static_list = [o for o in obstacle_list if "static.prop" in o.type_id]
        obstacle_state, obstacle, distance = self._vehicle_obstacle_detected(
            static_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=self.FORWARD_ANGLE_STRAIGHT)
        if obstacle_state and not self._is_obstacle_in_lane(obstacle, waypoint):
            return False, None, -1
        return obstacle_state, obstacle, distance

    def _update_stall_tracking(self, vehicle):
        """
        Tracks how long a specific vehicle ahead has been reporting
        near-zero speed, AND how long ego itself has also been essentially
        stationary while that vehicle is tracked. Both are required to
        distinguish a genuinely stalled/wrecked vehicle (AccidentTwoWays)
        from an ordinary lead vehicle that's simply queued in traffic (or
        hasn't been released by the traffic manager yet) while ego is
        still comfortably cruising or crawling forward behind it - that
        case is normal car-following territory, not a permanent blockage.

            :param vehicle: the vehicle currently detected ahead, or None
        """
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
            # A brand-new tracked vehicle starts fresh at 0 ticks: it must be
            # observed stationary for at least one full SUBSEQUENT tick before
            # any counting begins, rather than being credited a stall tick on
            # the very tick it is first seen. Without this early return, the
            # first sighting of any near-zero-speed vehicle immediately scored
            # tick 1 - an off-by-one that muddles "just spotted, not yet
            # observed over time" with "confirmed stationary long enough".
            return

        self._stalled_tick_counter += 1
        if self._speed <= self.EGO_BLOCKED_SPEED_THRESHOLD:
            self._ego_blocked_tick_counter += 1
        else:
            self._ego_blocked_tick_counter = 0

    def _stalled_vehicle_confirmed(self):
        """
        :return: True once the currently tracked vehicle has been
            stationary long enough, AND ego has itself been essentially
            stopped directly behind it for long enough, to be treated as
            a permanent obstacle worth a lane departure.

            Suppressed while ego has never yet driven at a normal pace
            since the episode began (`_ego_has_moved_normally`) AND the
            block hasn't lasted excessively long either
            (`STALL_STARTUP_HARD_BLOCK_TICKS`): at the very start of a
            route every actor, ego included, spawns at rest and takes a
            moment to accelerate for the first time - that is normal
            launch inertia, not a genuine stall/wreck, even though it can
            otherwise satisfy both timers below purely by the coincidence
            of ego being stuck behind a vehicle going through the exact
            same startup ramp-up. Using "has ego ever actually moved" (a
            one-way latch) rather than a fixed tick count means this
            adapts to however long that ramp-up genuinely takes, instead
            of guessing a fixed duration. `STALL_STARTUP_HARD_BLOCK_TICKS`
            is only a safety net for the rarer case of a real obstacle
            sitting at the route's very first waypoint.
        """
        if (not self._ego_has_moved_normally
                and self._ego_blocked_tick_counter <= self.STALL_STARTUP_HARD_BLOCK_TICKS):
            return False
        if self._tick_count <= self.STALL_STARTUP_GRACE_TICKS:
            return False
        return (self._stalled_tick_counter > self.STALL_TIMEOUT_TICKS
                and self._ego_blocked_tick_counter > self.EGO_BLOCKED_CONFIRM_TICKS)

    def _near_junction(self, location, distance):
        """
        True if a junction starts within `distance` metres ahead of
        `location` along its current lane, or if `location` already sits
        inside one. Used to tell a vehicle genuinely stalled in open road
        (AccidentTwoWays, a broken-down car) apart from one simply obeying
        a stop sign or a red light, or still in the first seconds of
        pulling away from either - all of which happen right at a
        junction and must never be mistaken for a permanent blockage,
        however long the wait lasts.

            :param location: the point to check
            :param distance: how far ahead to look, in metres
            :return: True if a junction is at or within reach of that point
        """
        wp = self._map.get_waypoint(location)
        if wp.is_junction:
            return True
        ahead = wp.next(distance)
        return any(w.is_junction for w in ahead)

    def _stalled_vehicle_ahead(self, waypoint):
        """
        Detects a vehicle ahead that has been confirmed stalled (see
        `_update_stall_tracking`): a wrecked or broken-down vehicle blocking
        the lane (AccidentTwoWays), as opposed to a static prop already
        covered by `_static_obstacle_ahead`.

        Deliberately excludes a cyclist (handled by `_cyclist_ahead`) and
        any vehicle stopped at or approaching a junction (see
        `_near_junction`) - a car paused for a stop sign, a red light, or
        just pulling away from either must never be diagnosed as stalled,
        no matter how long the wait (user request).

            :param waypoint: the agent's current waypoint
            :return: tuple (obstacle_state, obstacle, distance)
        """
        obstacle_list = self._build_obstacle_list(waypoint, max_distance=self.BYPASS_DETECTION_DISTANCE)
        vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
            obstacle_list, self.BYPASS_DETECTION_DISTANCE, up_angle_th=self.FORWARD_ANGLE_STRAIGHT)

        if vehicle_state and self._is_cyclist(vehicle):
            vehicle_state = False

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
        ever justified by one of exactly three situations (user request):
          1. a static obstacle (ConstructionObstacleTwoWays, roadworks...);
          2. a cyclist ahead or beside (see `_cyclist_ahead`);
          3. a vehicle confirmed genuinely stalled - NOT a car paused at a
             stop sign/red light/pulling away (see `_stalled_vehicle_ahead`).
        Anything else in front of the agent is plain car-following
        (`car_following_manager`), never a reason to change lane.

            :param waypoint: the agent's current waypoint
            :return: tuple (obstacle_state, obstacle, distance)
        """
        static_state, static_obstacle, static_distance = self._static_obstacle_ahead(waypoint)
        if static_state:
            return static_state, static_obstacle, static_distance

        cyclist_state, cyclist, cyclist_distance = self._cyclist_ahead(waypoint)
        if cyclist_state:
            return cyclist_state, cyclist, cyclist_distance

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

    def _estimate_obstacle_length(self, obstacle):
        """
        Rough longitudinal footprint of the obstacle to clear, from its
        bounding box (its largest horizontal extent, doubled).

            :param obstacle: the obstacle actor
            :return: approximate length (m)
        """
        return 2.0 * max(obstacle.bounding_box.extent.x, obstacle.bounding_box.extent.y)

    def _bypass_merge_point(self, waypoint, obstacle, obstacle_distance):
        """
        Waypoint, on the ORIGINAL lane, far enough past the obstacle to
        safely merge back.

            :param waypoint: the agent's current waypoint
            :param obstacle: the obstacle actor being bypassed
            :param obstacle_distance: current distance to the obstacle (m)
            :return: a waypoint on the original lane past the obstacle, or
                     None if the road doesn't extend that far (dead end,
                     map edge...)
        """
        overtake_length = obstacle_distance + self._estimate_obstacle_length(obstacle) + self.BYPASS_OVERTAKE_MARGIN
        ahead = waypoint.next(overtake_length)
        if not ahead:
            return None
        return ahead[0]

    def _start_bypass_maneuver(self, waypoint, obstacle, obstacle_distance):
        """
        Triggers a lateral shift to the opposite lane to bypass the
        obstacle, targeting a merge point genuinely past it rather than
        the very next waypoint (see `_bypass_merge_point`).

            :param waypoint: the agent's current waypoint
            :param obstacle: the obstacle actor being bypassed
            :param obstacle_distance: current distance to the obstacle (m)
            :return: True if the manoeuvre could be started, False if there
                     is no usable opposite lane or merge point at this
                     point (edge of the road, junction, map end...) - the
                     caller must then abandon the bypass instead of
                     crashing on a None waypoint.
        """
        opposite_wpt = waypoint.get_left_lane()
        if opposite_wpt is None or opposite_wpt.lane_type != carla.LaneType.Driving:
            return False
        merge_waypoint = self._bypass_merge_point(waypoint, obstacle, obstacle_distance)
        if merge_waypoint is None:
            return False
        self.set_destination(merge_waypoint.transform.location, opposite_wpt.transform.location)
        self._bypass_target_actor = obstacle
        return True

    def _ego_longitudinal_offset(self, actor):
        """
        Signed distance of `actor` ahead of (>0) or behind (<0) the ego,
        measured along the ego's OWN current forward vector rather than the
        road frame at a waypoint.

        `_road_projection` uses the heading of the ego's current road
        waypoint as its longitudinal axis. On a sharply curved road (and
        Town12 is very sinuous) that axis rotates every tick, so an obstacle
        the ego has physically driven past can still project to a positive
        "longitudinal" value and never read as "behind" - which is exactly
        why a cyclist overtake could hang until `BYPASS_TIMEOUT_TICKS`.
        Projecting onto the ego's actual forward vector instead ties the
        ahead/behind test to where the car is really pointing, independent
        of how the lane curves, so "I have passed it" is detected reliably
        even in the middle of a bend.

            :param actor: the actor to locate relative to the ego
            :return: longitudinal offset in metres (positive = ahead of ego)
        """
        ego_tf = self._vehicle.get_transform()
        forward = ego_tf.get_forward_vector()
        loc = actor.get_location()
        dx = loc.x - ego_tf.location.x
        dy = loc.y - ego_tf.location.y
        return dx * forward.x + dy * forward.y

    def _obstacle_cleared(self, waypoint):
        """
        Indicates whether the SPECIFIC obstacle being bypassed has now been
        passed, using its geometric position relative to the agent rather
        than re-running the same forward scanner used to detect it.

        The ahead/behind test is taken along the ego's actual heading (see
        `_ego_longitudinal_offset`) rather than the ego's current road
        waypoint frame: on a curved road the waypoint frame rotates each
        tick, which could keep a physically-passed obstacle reading as
        "still ahead" and hang the manoeuvre until it timed out. A
        `CLEAR_MARGIN` past zero avoids merging back while the obstacle is
        still level with the ego.

            :param waypoint: the agent's current waypoint
            :return: True if the tracked obstacle is now behind the agent
        """
        if self._bypass_target_actor is None:
            obstacle_state, _, _ = self._blocking_obstacle_ahead(waypoint)
            return not obstacle_state
        CLEAR_MARGIN = 5.0
        return self._ego_longitudinal_offset(self._bypass_target_actor) < -CLEAR_MARGIN

    def _update_bypass_target_resumed_tracking(self):
        """
        Tracks whether the vehicle currently being bypassed has sustained a
        normal driving speed during the 'overtaking' phase. A genuinely
        stalled/wrecked vehicle (AccidentTwoWays) stays essentially
        motionless throughout the manoeuvre; if it instead sustains a real
        speed, it was never actually stalled - just an ordinary lead
        vehicle that paused briefly in traffic (queue, TM not having
        released it yet...) and got misclassified as a permanent obstacle.
        Continuing to chase it would just be ordinary car-following
        disguised as an overtake (see `BYPASS_ABORT_RESUME_TICKS`).
        """
        if self._bypass_target_actor is None:
            self._bypass_target_resumed_tick_counter = 0
            return
        if get_speed(self._bypass_target_actor) > self.BYPASS_ABORT_RESUME_SPEED_THRESHOLD:
            self._bypass_target_resumed_tick_counter += 1
        else:
            self._bypass_target_resumed_tick_counter = 0

    def _bypass_target_resumed_normal_driving(self):
        """
        :return: True once the tracked obstacle has sustained a normal
            driving speed long enough during the manoeuvre to conclude it
            was never a genuine permanent blockage, and the bypass should
            be aborted rather than pursued until `BYPASS_TIMEOUT_TICKS`.
        """
        return self._bypass_target_resumed_tick_counter > self.BYPASS_ABORT_RESUME_TICKS

    def _back_on_original_lane(self, waypoint):
        """
        Indicates whether the agent has returned to its original path after taking a detour.

            :param waypoint: the agent’s current waypoint
            :return: True if the current path matches the starting path
        """
        return (self._bypass_origin_waypoint is not None
                and waypoint.lane_id == self._bypass_origin_waypoint.lane_id)

    def _record_bypass_failure(self, obstacle):
        """
        Tracks consecutive bypass failures (timeout / no usable opposite
        lane / target resumed normal driving) against the SAME obstacle
        actor. A genuinely bypassable obstacle (a real
        ConstructionObstacleTwoWays/AccidentTwoWays/ParkedObstacleTwoWays)
        normally succeeds within one or two attempts; looping forever
        against the very same actor is a sign this attempt wasn't a real,
        passable blockage right now - most likely an ordinary vehicle we
        are simply following (`_stalled_vehicle_confirmed` and the
        resumed-driving abort should now catch most of those), or a
        face-off deadlock our own manoeuvre created with it. Once the
        retry budget is exhausted we give up on that specific actor for
        `BYPASS_GIVEUP_COOLDOWN_TICKS` (see `bypass_obstacle_manager`) and
        fall back to plain car-following instead of retrying immediately -
        NOT for the rest of the episode, since the same actor can later
        become a genuine permanent blockage worth a fresh attempt.

            :param obstacle: the obstacle actor the failed attempt was
                against, or None
        """
        obstacle_id = obstacle.id if obstacle is not None else None
        if obstacle_id != self._last_bypass_failed_id:
            self._last_bypass_failed_id = obstacle_id
            self._bypass_retry_count = 0
        self._bypass_retry_count += 1
        if obstacle_id is not None and self._bypass_retry_count > self.BYPASS_MAX_CONSECUTIVE_RETRIES:
            self._bypass_giveup_id = obstacle_id
            self._bypass_giveup_tick = self._tick_count
            print(f"[BYPASS] Giving up on obstacle id={obstacle_id} after "
                  f"{self._bypass_retry_count} failed attempts -> reverting to car-following "
                  f"for {self.BYPASS_GIVEUP_COOLDOWN_TICKS} ticks")

    def _bypass_giveup_active(self, obstacle_id):
        """
        :param obstacle_id: id of the obstacle currently detected as
            blocking (or None)
        :return: True if this specific actor is still within its
            give-up cooldown window and should be ignored by the bypass
            module for now (see `BYPASS_GIVEUP_COOLDOWN_TICKS`).
        """
        if obstacle_id is None or obstacle_id != self._bypass_giveup_id:
            return False
        return (self._tick_count - self._bypass_giveup_tick) <= self.BYPASS_GIVEUP_COOLDOWN_TICKS

    def _reset_bypass_state(self):
        """
        Resets the status of the bypass module. Deliberately leaves the
        retry circuit-breaker bookkeeping (`_bypass_giveup_id`,
        `_last_bypass_failed_id`, `_bypass_retry_count`) untouched - see
        `_record_bypass_failure`.
        """
        self._bypass_state = 'idle'
        self._bypass_origin_waypoint = None
        self._bypass_start_tick = None
        self._last_bypass_transition_tick = None
        self._bypass_target_actor = None
        self._bypass_target_resumed_tick_counter = 0
        self._stalled_vehicle_id = None
        self._stalled_tick_counter = 0
        self._ego_blocked_tick_counter = 0

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
            obstacle_state, obstacle, _ = self._blocking_obstacle_ahead(waypoint)
            if obstacle_state and self._bypass_giveup_active(obstacle.id):
                obstacle_state = False
            if obstacle_state:
                self._bypass_origin_waypoint = waypoint
                self._bypass_target_actor = obstacle
                self._bypass_state = 'waiting_gap'
                self._bypass_start_tick = self._tick_count
                self._last_bypass_transition_tick = self._tick_count
                self._log_bypass_transition('detected', waypoint, obstacle=obstacle)
            return self._bypass_state != 'idle'

        if self._bypass_timed_out():
            self._log_bypass_transition('timeout', waypoint)
            self._record_bypass_failure(self._bypass_target_actor)
            self._reset_bypass_state()
            return False

        if not self._bypass_target_still_present(waypoint):
            self._log_bypass_transition('cyclist_gone', waypoint, obstacle=self._bypass_target_actor)
            self._reset_bypass_state()
            return False

        if self._bypass_state == 'waiting_gap':
            # Re-validate the memorized target BEFORE committing to the lane
            # change: if the vehicle we confirmed stalled has started rolling
            # again while we waited for a gap, the reason for overtaking is
            # gone. Abort now, before any lateral move, and revert to plain
            # car-following. Not flagged as a scenario failure (correct
            # self-diagnosis) and NOT fed to the give-up circuit-breaker: no
            # manoeuvre was wasted, and a genuine re-stall later still takes
            # a full STALL_TIMEOUT_TICKS to re-confirm, so this can't loop.
            if self._bypass_target_resumed_before_start():
                self._log_bypass_transition('resumed_normal', waypoint, obstacle=self._bypass_target_actor)
                self._reset_bypass_state()
                return False

            if self._can_start_bypass(waypoint):

                obstacle = self._bypass_target_actor

                if obstacle is None or not obstacle.is_alive:
                    self._reset_bypass_state()
                    return False

                longitudinal, _ = self._road_projection(obstacle, waypoint)

                if longitudinal <= 0:
                    self._reset_bypass_state()
                    return False

                obstacle_distance = longitudinal

                if self._start_bypass_maneuver(
                        waypoint,
                        obstacle,
                        obstacle_distance):

                    self._bypass_state = "overtaking"
                    self._last_bypass_transition_tick = self._tick_count
                    self._log_bypass_transition("gap_found", waypoint)

                else:
                    self._log_bypass_transition("no_lane", waypoint)
                    self._record_bypass_failure(obstacle)
                    self._reset_bypass_state()
                    return False

            return True

        if self._bypass_state == 'overtaking':
            self._update_bypass_target_resumed_tracking()
            if self._bypass_target_resumed_normal_driving():
                self._log_bypass_transition('resumed_normal', waypoint, obstacle=self._bypass_target_actor)
                self._record_bypass_failure(self._bypass_target_actor)
                self._reset_bypass_state()
                return False
            if self._obstacle_cleared(waypoint):
                self._bypass_state = 'returning'
                self._last_bypass_transition_tick = self._tick_count
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
        Indicates whether the agent has been waiting too long to enter or
        cross the junction (prevents the "stuck forever at an
        intersection" failure mode and the resulting scenario timeout
        infraction).

        Measured against `self._tick_count` rather than a counter
        incremented only while this method is called - see the identical
        fix and rationale on `_control_loss_timed_out` / `_bypass_timed_out`.

            :return: True if the wait is considered excessive
        """
        return (self._tick_count - self._junction_start_tick) > self.JUNCTION_TIMEOUT_TICKS

    def _reset_junction_state(self):
        """Resets the status of the junction-crossing module."""
        self._junction_state = 'idle'
        self._junction_start_tick = None

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

        # NOTE: this pre-filter used to be a fixed 10 m radius regardless of
        # speed - the same "hidden bottleneck" bug already fixed for vehicles
        # in _collision_detection_range(). At cruising speed (~46-49 km/h,
        # ~13 m/s) 10 m gives under a second of reaction time, which is
        # consistent with the pedestrian collision seen in simulation logs.
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

    def _road_heading(self, waypoint):
        """
        :param waypoint: the agent's current waypoint
        :return: the road's own heading (degrees) at that waypoint
        """
        return waypoint.transform.rotation.yaw

    def _velocity_heading(self):
        """
        :return: the vehicle's actual direction of travel (degrees), taken
            from its velocity vector rather than its bodywork orientation,
            so a sideways skid is visible even if the chassis still points
            forward.
        """
        vel = self._vehicle.get_velocity()
        return math.degrees(math.atan2(vel.y, vel.x))

    def _heading_deviation(self, waypoint):
        """
        Angular gap (0-180°) between the vehicle's actual velocity heading
        and the road heading at the CURRENT waypoint. Comparing against the
        current waypoint (which already follows the curve) rather than a
        fixed reference means a normal bend is not mistaken for a skid -
        only an uncommanded slide relative to the road right there shows
        up as a large gap.

            :param waypoint: the agent's current waypoint
            :return: absolute angular deviation in degrees
        """
        diff = (self._velocity_heading() - self._road_heading(waypoint)) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        return diff

    def _wet_severity(self):
        """
        0 (dry) to 1 (fully wet) read from the world's current weather, used
        to tighten the control-loss margins as the route's weather degrades
        (§4.2: "adhérence réduite ajoutée à la perte de contrôle simulée").
        Never raises: falls back to 0 (dry) if the weather can't be read,
        e.g. in unit tests that don't stub a world weather.

            :return: wetness severity between 0.0 and 1.0
        """
        try:
            weather = self._world.get_weather()
        except AttributeError:
            return 0.0
        wetness = getattr(weather, 'wetness', 0.0)
        precipitation = getattr(weather, 'precipitation', 0.0)
        return max(wetness, precipitation) / 100.0

    def _control_loss_heading_threshold(self):
        """
        :return: CONTROL_LOSS_HEADING_THRESHOLD, narrowed towards
            WET_HEADING_MARGIN as the weather gets wetter (so the agent
            reacts to a smaller deviation once grip is already reduced).
        """
        severity = self._wet_severity()
        margin = 1.0 - (1.0 - self.WET_HEADING_MARGIN) * severity
        return self.CONTROL_LOSS_HEADING_THRESHOLD * margin

    def _control_loss_stabilize_speed(self):
        """
        :return: CONTROL_LOSS_STABILIZE_SPEED, lowered towards
            WET_SPEED_MARGIN as the weather gets wetter.
        """
        severity = self._wet_severity()
        margin = 1.0 - (1.0 - self.WET_SPEED_MARGIN) * severity
        return self.CONTROL_LOSS_STABILIZE_SPEED * margin

    def _in_bypass_maneuver_grace_period(self):
        """
        True for a short window right after the bypass module has just
        commanded a lane change (`_start_bypass_maneuver`, or the
        'overtaking' -> 'returning' transition back to the original lane).

            :return: True if control-loss detection should be suppressed
        """
        if self._bypass_state == 'idle' or self._last_bypass_transition_tick is None:
            return False
        return (self._tick_count - self._last_bypass_transition_tick) <= self.CONTROL_LOSS_BYPASS_GRACE_TICKS

    def _control_loss_detected(self, waypoint):
        """
        :param waypoint: the agent's current waypoint
        :return: True if the vehicle's heading has drifted away from the
            road by more than the (weather-adjusted) threshold, at a speed
            high enough for heading to be meaningful, and while the agent
            isn't in the middle of a DELIBERATE lane change (bypass
            overtake/return, tailgating shift...). Those manoeuvres cause a
            large, intentional heading gap by design - without this guard
            they were being misread as a skid (observed in simulation: a
            false "ControlLoss" during ordinary lane-change traffic caused
            the agent to lose the bypass in progress and drift off-road).
        """
        if self._direction in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            return False
        if self._in_bypass_maneuver_grace_period():
            return False
        if self._speed < self.CONTROL_LOSS_MIN_SPEED_KMH:
            return False
        return self._heading_deviation(waypoint) > self._control_loss_heading_threshold()

    def _control_loss_recovered(self, waypoint):
        """
        :param waypoint: the agent's current waypoint
        :return: True once the heading deviation has fallen back under the
            (lower) recovery threshold - the hysteresis gap between
            trigger and recovery avoids flapping in and out of the state
            on borderline readings.
        """
        return self._heading_deviation(waypoint) < self.CONTROL_LOSS_RECOVERY_THRESHOLD

    def _control_loss_timed_out(self):
        """
        :return: True once stabilization has been active for longer than
            CONTROL_LOSS_TIMEOUT_TICKS, so a stuck reading can't cap the
            vehicle's speed forever.
        """
        return (self._tick_count - self._control_loss_start_tick) > self.CONTROL_LOSS_TIMEOUT_TICKS

    def _reset_control_loss_state(self):
        """Resets the status of the control-loss stabilization module."""
        self._control_loss_state = 'idle'
        self._control_loss_start_tick = None
        self._control_loss_pending_ticks = 0

    def _log_control_loss_transition(self, event, waypoint=None):
        """
        Single entry point for control-loss log messages, mirroring
        `_log_bypass_transition` / `_log_junction_transition`.

            :param event: 'detected' | 'recovered' | 'timeout'
            :param waypoint: the agent's current waypoint (optional)
        """
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

    def control_loss_manager(self, waypoint):
        """
        Generic stabilization module for `ControlLoss` (§3.8/§6.2): detects
        an uncommanded heading deviation (skid) and, while it persists,
        caps cruise speed to a cautious value using THROTTLE ONLY - never
        `emergency_stop`'s hard brake, since braking hard mid-skid tends to
        make it worse. Steering itself is left to the local planner /
        lateral controller, which already corrects progressively rather
        than snapping back (see controller.py's steering-rate limiting) -
        this module's job is only to not fight that recovery with an
        abrupt braking input.

            :param waypoint: the agent's current waypoint
            :return: True while a stabilization response is in progress
                     (caller should apply the reduced cruise speed instead
                     of its normal behaviour)
        """
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
            blocking_control = self._react_to_moving_obstacle(
                ego_vehicle_wp, debug=debug,
                ignore_actor_id=self._bypass_target_actor.id if self._bypass_target_actor is not None else None)
            if blocking_control is not None:
                return blocking_control

            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(self._clamp_to_speed_limit(target_speed))
            return self._local_planner.run_step(debug=debug)

        # 2.3: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

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