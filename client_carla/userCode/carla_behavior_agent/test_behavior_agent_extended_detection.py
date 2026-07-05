# -*- coding: utf-8 -*-
"""
Tests unitaires pour le Bloc 1 : Détection élargie des obstacles "véhicule/objet".

Ces tests ne nécessitent PAS CARLA ni la simulation : les modules CARLA
(`carla`, `basic_agent`, `local_planner`, `behavior_types`, `misc`) sont
remplacés par de petits stubs qui reproduisent uniquement l'interface dont
`behavior_agent.py` a besoin. `BehaviorAgent` est instancié sans passer par
`__init__` (qui nécessite un vrai monde CARLA), afin de tester chaque
nouvelle fonction de manière isolée.
"""

import sys
import types
import importlib.util
import os
import unittest


# ---------------------------------------------------------------------------
# 1. Stubs des modules externes (carla, basic_agent, local_planner, ...)
# ---------------------------------------------------------------------------

def _install_stub_modules():
    # --- carla ---
    carla_stub = types.ModuleType("carla")

    class VehicleControl:
        def __init__(self):
            self.throttle = 0.0
            self.brake = 0.0
            self.hand_brake = False
            self.steer = 0.0

    class _LaneChange:
        NONE = 0
        Right = 1
        Left = 2
        Both = 3

    class _LaneType:
        Driving = 1
        Any = 2

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

        def __add__(self, other):
            return Location(self.x + other.x, self.y + other.y, self.z + other.z)

        def __sub__(self, other):
            return Location(self.x - other.x, self.y - other.y, self.z - other.z)

        def distance(self, other):
            return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    carla_stub.VehicleControl = VehicleControl
    carla_stub.LaneChange = _LaneChange
    carla_stub.LaneType = _LaneType
    carla_stub.Location = Location
    sys.modules["carla"] = carla_stub

    # --- basic_agent ---
    basic_agent_stub = types.ModuleType("basic_agent")

    class BasicAgent:
        def __init__(self, vehicle, opt_dict=None, map_inst=None, grp_inst=None):
            # Version minimale : la vraie classe configure world/map/local_planner,
            # ce qui n'est pas nécessaire pour les tests unitaires du bloc 1.
            self._vehicle = vehicle
            self._world = None
            self._map = map_inst
            self._local_planner = None
            self._max_brake = 0.5

        def _vehicle_obstacle_detected(self, *args, **kwargs):
            raise NotImplementedError("A stubber avec un mock dans le test.")

        def _affected_by_traffic_light(self, *args, **kwargs):
            return False, None

    basic_agent_stub.BasicAgent = BasicAgent
    sys.modules["basic_agent"] = basic_agent_stub

    # --- local_planner ---
    local_planner_stub = types.ModuleType("local_planner")

    class RoadOption:
        VOID = -1
        LEFT = 1
        RIGHT = 2
        STRAIGHT = 3
        LANEFOLLOW = 4
        CHANGELANELEFT = 5
        CHANGELANERIGHT = 6

    local_planner_stub.RoadOption = RoadOption
    sys.modules["local_planner"] = local_planner_stub

    # --- behavior_types ---
    behavior_types_stub = types.ModuleType("behavior_types")

    class _BaseBehavior:
        min_proximity_threshold = 10
        speed_decrease = 10
        max_speed = 50
        speed_lim_dist = 3
        safety_time = 3
        braking_distance = 5
        tailgate_counter = 0

    class Cautious(_BaseBehavior):
        pass

    class Normal(_BaseBehavior):
        pass

    class Aggressive(_BaseBehavior):
        pass

    behavior_types_stub.Cautious = Cautious
    behavior_types_stub.Normal = Normal
    behavior_types_stub.Aggressive = Aggressive
    sys.modules["behavior_types"] = behavior_types_stub

    # --- misc ---
    misc_stub = types.ModuleType("misc")

    def get_speed(actor):
        return getattr(actor, "speed", 0)

    def positive(value):
        return max(0.0, value)

    def is_within_distance(*args, **kwargs):
        return True

    def compute_distance(loc1, loc2):
        return loc1.distance(loc2)

    misc_stub.get_speed = get_speed
    misc_stub.positive = positive
    misc_stub.is_within_distance = is_within_distance
    misc_stub.compute_distance = compute_distance
    sys.modules["misc"] = misc_stub


_install_stub_modules()


def _load_behavior_agent_module():
    """Charge behavior_agent.py depuis le même dossier que ce test."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "behavior_agent.py")
    spec = importlib.util.spec_from_file_location("behavior_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


behavior_agent_module = _load_behavior_agent_module()
BehaviorAgent = behavior_agent_module.BehaviorAgent
RoadOption = sys.modules["local_planner"].RoadOption
carla = sys.modules["carla"]


# ---------------------------------------------------------------------------
# 2. Doubles de test (fake actors / waypoints)
# ---------------------------------------------------------------------------

class FakeActor:
    """Reproduit l'interface minimale d'un acteur CARLA (véhicule, cycliste, objet statique)."""

    _next_id = 0

    def __init__(self, location, type_id="vehicle.tesla.model3", speed=0):
        FakeActor._next_id += 1
        self.id = FakeActor._next_id
        self._location = location
        self.type_id = type_id
        self.speed = speed

    def get_location(self):
        return self._location


class FakeActorList(list):
    """Reproduit `world.get_actors()` : supporte `.filter(pattern)`."""

    def filter(self, pattern):
        pattern = pattern.strip("*")
        return FakeActorList([a for a in self if pattern in a.type_id])


class FakeWorld:
    def __init__(self, actors):
        self._actors = FakeActorList(actors)

    def get_actors(self):
        return self._actors


class FakeTransform:
    def __init__(self, location):
        self.location = location


class FakeWaypoint:
    def __init__(self, location, is_junction=False):
        self.transform = FakeTransform(location)
        self.is_junction = is_junction


def make_agent(world_actors, ego_location=None, incoming_direction=RoadOption.LANEFOLLOW,
               direction=RoadOption.LANEFOLLOW, speed_limit=50, speed=20):
    """
    Construit un BehaviorAgent minimal SANS passer par __init__ (qui exige un
    vrai monde CARLA), afin de tester les fonctions du bloc 1 isolément.
    """
    ego_location = ego_location or carla.Location(0, 0, 0)
    ego_vehicle = FakeActor(ego_location, type_id="vehicle.ego.car")

    agent = object.__new__(BehaviorAgent)
    agent._vehicle = ego_vehicle
    agent._world = FakeWorld(world_actors)
    agent._actors = agent._world.get_actors()
    agent._behavior = sys.modules["behavior_types"].Normal()
    agent._speed_limit = speed_limit
    agent._speed = speed
    agent._direction = direction
    agent._incoming_direction = incoming_direction
    agent._bypass_state = 'idle'
    agent._bypass_origin_waypoint = None
    agent._bypass_tick_counter = 0
    agent._tick_count = 0
    agent._scenario_result = None
    agent._junction_state = 'idle'
    agent._junction_tick_counter = 0
    return agent


# ---------------------------------------------------------------------------
# 3. Tests
# ---------------------------------------------------------------------------

class TestIsInTurn(unittest.TestCase):
    def test_true_when_incoming_direction_is_left(self):
        agent = make_agent([], incoming_direction=RoadOption.LEFT)
        self.assertTrue(agent._is_in_turn())

    def test_true_when_incoming_direction_is_right(self):
        agent = make_agent([], incoming_direction=RoadOption.RIGHT)
        self.assertTrue(agent._is_in_turn())

    def test_false_when_lanefollow(self):
        agent = make_agent([], incoming_direction=RoadOption.LANEFOLLOW)
        self.assertFalse(agent._is_in_turn())

    def test_false_when_straight(self):
        agent = make_agent([], incoming_direction=RoadOption.STRAIGHT)
        self.assertFalse(agent._is_in_turn())


class TestForwardDetectionAngle(unittest.TestCase):
    def test_widened_angle_in_turn(self):
        agent = make_agent([], incoming_direction=RoadOption.LEFT)
        self.assertEqual(agent._forward_detection_angle(), BehaviorAgent.FORWARD_ANGLE_TURN)

    def test_narrow_angle_in_straight_line(self):
        agent = make_agent([], incoming_direction=RoadOption.LANEFOLLOW)
        self.assertEqual(agent._forward_detection_angle(), BehaviorAgent.FORWARD_ANGLE_STRAIGHT)

    def test_turn_angle_is_wider_than_straight_angle(self):
        # Garde-fou : le bloc 1 doit élargir la détection en virage, jamais la réduire.
        self.assertGreater(BehaviorAgent.FORWARD_ANGLE_TURN, BehaviorAgent.FORWARD_ANGLE_STRAIGHT)


class TestBuildObstacleList(unittest.TestCase):
    def test_includes_cyclist_as_vehicle_actor(self):
        cyclist = FakeActor(carla.Location(10, 0, 0), type_id="vehicle.bh.crossbike")
        agent = make_agent([cyclist])
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        obstacles = agent._build_obstacle_list(waypoint)

        self.assertIn(cyclist, obstacles)

    def test_includes_static_prop_for_dynamic_object_crossing(self):
        container = FakeActor(carla.Location(15, 0, 0), type_id="static.prop.container")
        agent = make_agent([container])
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        obstacles = agent._build_obstacle_list(waypoint)

        self.assertIn(container, obstacles)

    def test_excludes_ego_vehicle(self):
        agent = make_agent([])
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))
        # L'ego véhicule est ajouté dans world_actors pour vérifier qu'il est bien exclu.
        agent._world._actors.append(agent._vehicle)

        obstacles = agent._build_obstacle_list(waypoint)

        self.assertNotIn(agent._vehicle, obstacles)

    def test_excludes_obstacles_beyond_max_distance(self):
        far_vehicle = FakeActor(carla.Location(1000, 0, 0), type_id="vehicle.tesla.model3")
        agent = make_agent([far_vehicle])
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        obstacles = agent._build_obstacle_list(waypoint)

        self.assertNotIn(far_vehicle, obstacles)

    def test_ignores_unrelated_actor_types(self):
        walker = FakeActor(carla.Location(5, 0, 0), type_id="walker.pedestrian.0001")
        agent = make_agent([walker])
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        obstacles = agent._build_obstacle_list(waypoint)

        self.assertNotIn(walker, obstacles)

    def test_respects_custom_max_distance(self):
        vehicle = FakeActor(carla.Location(20, 0, 0), type_id="vehicle.tesla.model3")
        agent = make_agent([vehicle])
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        self.assertNotIn(vehicle, agent._build_obstacle_list(waypoint, max_distance=10))
        self.assertIn(vehicle, agent._build_obstacle_list(waypoint, max_distance=30))


class TestLaneChangeObstacleDetected(unittest.TestCase):
    def test_calls_vehicle_obstacle_detected_with_wide_angle_and_lane_offset(self):
        calls = {}

        def fake_detect(vehicle_list, distance, up_angle_th=0, lane_offset=0, low_angle_th=0):
            calls["up_angle_th"] = up_angle_th
            calls["lane_offset"] = lane_offset
            return True, "some_vehicle", 12.0

        agent = make_agent([])
        agent._vehicle_obstacle_detected = fake_detect

        state, vehicle, distance = agent._lane_change_obstacle_detected([], lane_offset=-1)

        self.assertEqual(calls["up_angle_th"], 180)
        self.assertEqual(calls["lane_offset"], -1)
        self.assertTrue(state)
        self.assertEqual(distance, 12.0)


class TestForwardObstacleDetected(unittest.TestCase):
    def test_uses_widened_angle_when_in_turn(self):
        calls = {}

        def fake_detect(vehicle_list, distance, up_angle_th=0, low_angle_th=0, lane_offset=0):
            calls["up_angle_th"] = up_angle_th
            return False, None, -1

        agent = make_agent([], incoming_direction=RoadOption.LEFT)
        agent._vehicle_obstacle_detected = fake_detect

        agent._forward_obstacle_detected([])

        self.assertEqual(calls["up_angle_th"], BehaviorAgent.FORWARD_ANGLE_TURN)

    def test_uses_narrow_angle_in_straight_line(self):
        calls = {}

        def fake_detect(vehicle_list, distance, up_angle_th=0, low_angle_th=0, lane_offset=0):
            calls["up_angle_th"] = up_angle_th
            return False, None, -1

        agent = make_agent([], incoming_direction=RoadOption.LANEFOLLOW)
        agent._vehicle_obstacle_detected = fake_detect

        agent._forward_obstacle_detected([])

        self.assertEqual(calls["up_angle_th"], BehaviorAgent.FORWARD_ANGLE_STRAIGHT)


class TestCollisionAndCarAvoidManagerIntegration(unittest.TestCase):
    """Vérifie l'orchestration globale de la méthode publique après refactor."""

    def test_detects_cyclist_ahead_in_turn(self):
        cyclist = FakeActor(carla.Location(5, 0, 0), type_id="vehicle.bh.crossbike", speed=8)
        agent = make_agent([cyclist], incoming_direction=RoadOption.LEFT, direction=RoadOption.LANEFOLLOW)

        def fake_detect(vehicle_list, distance, up_angle_th=0, low_angle_th=0, lane_offset=0):
            # Simule une détection positive dès lors que l'angle élargi est utilisé.
            detected = up_angle_th == BehaviorAgent.FORWARD_ANGLE_TURN and cyclist in vehicle_list
            return (True, cyclist, 5.0) if detected else (False, None, -1)

        agent._vehicle_obstacle_detected = fake_detect
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        state, vehicle, distance = agent.collision_and_car_avoid_manager(waypoint)

        self.assertTrue(state)
        self.assertEqual(vehicle, cyclist)

    def test_lane_change_left_uses_lane_offset(self):
        agent = make_agent([], direction=RoadOption.CHANGELANELEFT)

        calls = {}

        def fake_detect(vehicle_list, distance, up_angle_th=0, low_angle_th=0, lane_offset=0):
            calls["lane_offset"] = lane_offset
            return False, None, -1

        agent._vehicle_obstacle_detected = fake_detect
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        agent.collision_and_car_avoid_manager(waypoint)

        self.assertEqual(calls["lane_offset"], -1)

    def test_no_tailgating_check_during_lane_change(self):
        # `_tailgating` ne doit être appelé que dans la branche "tout droit".
        agent = make_agent([], direction=RoadOption.CHANGELANERIGHT)
        agent._vehicle_obstacle_detected = lambda *a, **k: (False, None, -1)

        called = {"tailgate": False}

        def fake_tailgating(*args, **kwargs):
            called["tailgate"] = True

        agent._tailgating = fake_tailgating
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        agent.collision_and_car_avoid_manager(waypoint)

        self.assertFalse(called["tailgate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)