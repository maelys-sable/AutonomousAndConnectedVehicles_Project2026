# -*- coding: utf-8 -*-
"""
Tests unitaires pour le Bloc 2 : Module de contournement générique.

Couvre ConstructionObstacleTwoWays, AccidentTwoWays et ParkedObstacleTwoWays
(même mécanisme : détection d'obstacle statique -> recherche de créneau dans
le trafic opposé -> contournement -> retour sur voie).

Comme pour le bloc 1, ces tests stubent CARLA et ses modules dépendants pour
tourner sans simulation. Réutilise les mêmes doubles de test (FakeActor,
FakeWorld, FakeWaypoint...) que test_behavior_agent_block1.py.

Lancement :
    python -m pytest test_behavior_agent_block2.py -v
"""

import sys
import os
import unittest

# Réutilise l'installation des stubs et les doubles de test du bloc 1.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_behavior_agent_extended_detection import (
    FakeActor, FakeWorld, FakeWaypoint, FakeTransform, make_agent,
    BehaviorAgent, RoadOption, carla,
)


class FakeLaneWaypoint(FakeWaypoint):
    """FakeWaypoint enrichi avec lane_id et get_left_lane, pour le bloc 2."""

    def __init__(self, location, lane_id=1, is_junction=False):
        super().__init__(location, is_junction=is_junction)
        self.lane_id = lane_id
        self._left_lane = None

    def set_left_lane(self, waypoint):
        self._left_lane = waypoint

    def get_left_lane(self):
        return self._left_lane


class FakeLocalPlanner:
    def __init__(self, target_waypoint):
        self.target_waypoint = target_waypoint
        self.speed = None

    def set_speed(self, speed):
        self.speed = speed

    def run_step(self, debug=False):
        return "CONTROL"


# ---------------------------------------------------------------------------
# _gap_is_safe : logique pure, sans dépendance CARLA
# ---------------------------------------------------------------------------

class TestGapIsSafe(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent([])

    def test_no_oncoming_vehicle_is_safe(self):
        self.assertTrue(self.agent._gap_is_safe(oncoming_distance=-1, oncoming_speed=0))
        self.assertTrue(self.agent._gap_is_safe(oncoming_distance=None, oncoming_speed=50))

    def test_stationary_oncoming_vehicle_is_safe(self):
        self.assertTrue(self.agent._gap_is_safe(oncoming_distance=10, oncoming_speed=0))

    def test_large_gap_is_safe(self):
        # 100 m à 36 km/h (10 m/s) -> 10 s d'arrivée, largement > BYPASS_MIN_GAP_TIME (4 s)
        self.assertTrue(self.agent._gap_is_safe(oncoming_distance=100, oncoming_speed=36))

    def test_small_gap_is_unsafe(self):
        # 20 m à 36 km/h (10 m/s) -> 2 s d'arrivée, < BYPASS_MIN_GAP_TIME (4 s)
        self.assertFalse(self.agent._gap_is_safe(oncoming_distance=20, oncoming_speed=36))

    def test_boundary_gap_is_safe(self):
        # Exactement au seuil : 40 m à 36 km/h (10 m/s) -> 4 s pile
        self.assertTrue(self.agent._gap_is_safe(oncoming_distance=40, oncoming_speed=36))


# ---------------------------------------------------------------------------
# _can_start_bypass : combine _oncoming_lane_obstacle + _gap_is_safe
# ---------------------------------------------------------------------------

class TestCanStartBypass(unittest.TestCase):
    def test_true_when_no_oncoming_vehicle(self):
        agent = make_agent([])
        agent._oncoming_lane_obstacle = lambda wp: (False, None, -1)
        self.assertTrue(agent._can_start_bypass(FakeWaypoint(carla.Location(0, 0, 0))))

    def test_true_when_gap_is_safe(self):
        oncoming = FakeActor(carla.Location(100, 0, 0), type_id="vehicle.tesla.model3", speed=36)
        agent = make_agent([])
        agent._oncoming_lane_obstacle = lambda wp: (True, oncoming, 100)
        self.assertTrue(agent._can_start_bypass(FakeWaypoint(carla.Location(0, 0, 0))))

    def test_false_when_gap_is_unsafe(self):
        oncoming = FakeActor(carla.Location(20, 0, 0), type_id="vehicle.tesla.model3", speed=36)
        agent = make_agent([])
        agent._oncoming_lane_obstacle = lambda wp: (True, oncoming, 20)
        self.assertFalse(agent._can_start_bypass(FakeWaypoint(carla.Location(0, 0, 0))))


# ---------------------------------------------------------------------------
# _static_obstacle_ahead : ne doit considérer que les objets statiques
# ---------------------------------------------------------------------------

class TestStaticObstacleAhead(unittest.TestCase):
    def test_filters_out_vehicles_keeps_static_props(self):
        cone_zone = FakeActor(carla.Location(20, 0, 0), type_id="static.prop.constructioncone")
        other_car = FakeActor(carla.Location(15, 0, 0), type_id="vehicle.tesla.model3")
        agent = make_agent([cone_zone, other_car])
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        seen_lists = {}

        def fake_detect(obstacle_list, distance, up_angle_th=0, low_angle_th=0, lane_offset=0):
            seen_lists["list"] = obstacle_list
            return (True, obstacle_list[0], 20.0) if obstacle_list else (False, None, -1)

        agent._vehicle_obstacle_detected = fake_detect
        state, obstacle, distance = agent._static_obstacle_ahead(waypoint)

        self.assertIn(cone_zone, seen_lists["list"])
        self.assertNotIn(other_car, seen_lists["list"])
        self.assertTrue(state)

    def test_no_obstacle_returns_false(self):
        agent = make_agent([])
        agent._vehicle_obstacle_detected = lambda *a, **k: (False, None, -1)
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))
        state, _, _ = agent._static_obstacle_ahead(waypoint)
        self.assertFalse(state)


# ---------------------------------------------------------------------------
# _obstacle_cleared / _back_on_original_lane : conditions de sortie du cycle
# ---------------------------------------------------------------------------

class TestObstacleClearedAndLaneReturn(unittest.TestCase):
    def test_obstacle_cleared_true_when_no_longer_detected(self):
        agent = make_agent([])
        agent._static_obstacle_ahead = lambda wp: (False, None, -1)
        self.assertTrue(agent._obstacle_cleared(FakeWaypoint(carla.Location(0, 0, 0))))

    def test_obstacle_cleared_false_while_still_ahead(self):
        obstacle = FakeActor(carla.Location(5, 0, 0), type_id="static.prop.container")
        agent = make_agent([])
        agent._static_obstacle_ahead = lambda wp: (True, obstacle, 5.0)
        self.assertFalse(agent._obstacle_cleared(FakeWaypoint(carla.Location(0, 0, 0))))

    def test_back_on_original_lane_true_when_lane_id_matches(self):
        agent = make_agent([])
        agent._bypass_origin_waypoint = FakeLaneWaypoint(carla.Location(0, 0, 0), lane_id=3)
        current = FakeLaneWaypoint(carla.Location(50, 0, 0), lane_id=3)
        self.assertTrue(agent._back_on_original_lane(current))

    def test_back_on_original_lane_false_when_still_on_opposite_lane(self):
        agent = make_agent([])
        agent._bypass_origin_waypoint = FakeLaneWaypoint(carla.Location(0, 0, 0), lane_id=3)
        current = FakeLaneWaypoint(carla.Location(30, 0, 0), lane_id=-3)
        self.assertFalse(agent._back_on_original_lane(current))

    def test_back_on_original_lane_false_when_no_origin_recorded(self):
        agent = make_agent([])
        agent._bypass_origin_waypoint = None
        current = FakeLaneWaypoint(carla.Location(0, 0, 0), lane_id=3)
        self.assertFalse(agent._back_on_original_lane(current))


# ---------------------------------------------------------------------------
# _start_bypass_maneuver : déclenche bien le changement de destination
# ---------------------------------------------------------------------------

class TestStartBypassManeuver(unittest.TestCase):
    def test_sets_destination_towards_left_lane(self):
        origin = FakeLaneWaypoint(carla.Location(0, 0, 0), lane_id=1)
        left_lane = FakeLaneWaypoint(carla.Location(0, 3, 0), lane_id=-1)
        origin.set_left_lane(left_lane)
        target = FakeLaneWaypoint(carla.Location(100, 0, 0), lane_id=1)

        agent = make_agent([])
        agent._local_planner = FakeLocalPlanner(target_waypoint=target)

        calls = {}

        def fake_set_destination(end_loc, start_loc):
            calls["end"] = end_loc
            calls["start"] = start_loc

        agent.set_destination = fake_set_destination
        agent._start_bypass_maneuver(origin)

        self.assertEqual(calls["end"], target.transform.location)
        self.assertEqual(calls["start"], left_lane.transform.location)


# ---------------------------------------------------------------------------
# bypass_obstacle_manager : cycle complet de la machine à états
# ---------------------------------------------------------------------------

class TestBypassObstacleManagerStateMachine(unittest.TestCase):
    def _agent_with_stubbed_steps(self):
        agent = make_agent([])
        agent._static_obstacle_ahead = lambda wp: (False, None, -1)
        agent._can_start_bypass = lambda wp: False
        agent._start_bypass_maneuver = lambda wp: None
        agent._obstacle_cleared = lambda wp: False
        agent._back_on_original_lane = lambda wp: False
        return agent

    def test_idle_stays_idle_without_obstacle(self):
        agent = self._agent_with_stubbed_steps()
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))
        self.assertFalse(agent.bypass_obstacle_manager(waypoint))
        self.assertEqual(agent._bypass_state, 'idle')

    def test_idle_to_waiting_gap_when_obstacle_detected(self):
        agent = self._agent_with_stubbed_steps()
        agent._static_obstacle_ahead = lambda wp: (True, FakeActor(carla.Location(20, 0, 0)), 20.0)
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        result = agent.bypass_obstacle_manager(waypoint)

        self.assertTrue(result)
        self.assertEqual(agent._bypass_state, 'waiting_gap')
        self.assertIs(agent._bypass_origin_waypoint, waypoint)

    def test_waiting_gap_stays_until_safe(self):
        agent = self._agent_with_stubbed_steps()
        agent._bypass_state = 'waiting_gap'
        agent._can_start_bypass = lambda wp: False

        result = agent.bypass_obstacle_manager(FakeWaypoint(carla.Location(0, 0, 0)))

        self.assertTrue(result)
        self.assertEqual(agent._bypass_state, 'waiting_gap')

    def test_waiting_gap_to_overtaking_when_safe(self):
        agent = self._agent_with_stubbed_steps()
        agent._bypass_state = 'waiting_gap'
        agent._can_start_bypass = lambda wp: True
        maneuver_calls = []
        agent._start_bypass_maneuver = lambda wp: maneuver_calls.append(wp)

        result = agent.bypass_obstacle_manager(FakeWaypoint(carla.Location(0, 0, 0)))

        self.assertTrue(result)
        self.assertEqual(agent._bypass_state, 'overtaking')
        self.assertEqual(len(maneuver_calls), 1)

    def test_overtaking_to_returning_once_obstacle_cleared(self):
        agent = self._agent_with_stubbed_steps()
        agent._bypass_state = 'overtaking'
        agent._obstacle_cleared = lambda wp: True

        result = agent.bypass_obstacle_manager(FakeWaypoint(carla.Location(0, 0, 0)))

        self.assertTrue(result)
        self.assertEqual(agent._bypass_state, 'returning')

    def test_overtaking_stays_while_obstacle_still_ahead(self):
        agent = self._agent_with_stubbed_steps()
        agent._bypass_state = 'overtaking'
        agent._obstacle_cleared = lambda wp: False

        result = agent.bypass_obstacle_manager(FakeWaypoint(carla.Location(0, 0, 0)))

        self.assertTrue(result)
        self.assertEqual(agent._bypass_state, 'overtaking')

    def test_returning_resets_to_idle_once_back_on_lane(self):
        agent = self._agent_with_stubbed_steps()
        agent._bypass_state = 'returning'
        agent._bypass_origin_waypoint = FakeWaypoint(carla.Location(0, 0, 0))
        agent._back_on_original_lane = lambda wp: True

        result = agent.bypass_obstacle_manager(FakeWaypoint(carla.Location(0, 0, 0)))

        self.assertFalse(result)
        self.assertEqual(agent._bypass_state, 'idle')
        self.assertIsNone(agent._bypass_origin_waypoint)

    def test_returning_stays_until_back_on_lane(self):
        agent = self._agent_with_stubbed_steps()
        agent._bypass_state = 'returning'
        agent._back_on_original_lane = lambda wp: False

        result = agent.bypass_obstacle_manager(FakeWaypoint(carla.Location(0, 0, 0)))

        self.assertTrue(result)
        self.assertEqual(agent._bypass_state, 'returning')

    def test_full_cycle_idle_to_idle(self):
        """Parcourt les quatre états dans l'ordre, comme lors d'un contournement réel."""
        agent = self._agent_with_stubbed_steps()
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        # idle -> waiting_gap (obstacle détecté)
        agent._static_obstacle_ahead = lambda wp: (True, FakeActor(carla.Location(20, 0, 0)), 20.0)
        self.assertTrue(agent.bypass_obstacle_manager(waypoint))
        self.assertEqual(agent._bypass_state, 'waiting_gap')

        # waiting_gap -> overtaking (créneau sûr)
        agent._can_start_bypass = lambda wp: True
        self.assertTrue(agent.bypass_obstacle_manager(waypoint))
        self.assertEqual(agent._bypass_state, 'overtaking')

        # overtaking -> returning (obstacle dépassé)
        agent._obstacle_cleared = lambda wp: True
        self.assertTrue(agent.bypass_obstacle_manager(waypoint))
        self.assertEqual(agent._bypass_state, 'returning')

        # returning -> idle (voie d'origine retrouvée)
        agent._back_on_original_lane = lambda wp: True
        self.assertFalse(agent.bypass_obstacle_manager(waypoint))
        self.assertEqual(agent._bypass_state, 'idle')


if __name__ == "__main__":
    unittest.main(verbosity=2)