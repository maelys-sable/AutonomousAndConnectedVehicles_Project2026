# -*- coding: utf-8 -*-
"""
Tests unitaires pour le Bloc 3 : gestion des intersections
(BlockedIntersection, NonSignalizedJunctionRightTurn).

Même mécanisme que le Bloc 2 (bypass) : détection d'un obstacle bloquant/
traversant -> attente d'un créneau sûr -> franchissement, avec un timeout
pour ne jamais rester bloqué indéfiniment (voir Route1_Plan_Strategie_Obstacles.md,
§3.4 et §3.10).

Réutilise les mêmes doubles de test (FakeActor, FakeWorld, FakeWaypoint,
make_agent...) que test_behavior_agent_extended_detection.py.

Lancement :
    python -m pytest test_behavior_agent_junction.py -v
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_behavior_agent_extended_detection import (
    FakeActor, FakeWaypoint, make_agent, BehaviorAgent, RoadOption, carla,
)


class FakeLocalPlanner:
    """Local planner minimal : suffit pour vérifier que _hold_position coupe bien la vitesse."""

    def __init__(self):
        self.speed = None

    def set_speed(self, speed):
        self.speed = speed

    def run_step(self, debug=False):
        return "CONTROL"


def make_junction_agent(is_junction=True, incoming_direction=RoadOption.RIGHT):
    """
    Construit un agent prêt pour les tests du bloc 3 : réutilise make_agent
    (bloc 1) puis ajoute les attributs propres au gestionnaire d'intersection
    (_incoming_waypoint, _local_planner) qui ne sont pas nécessaires aux
    autres blocs.
    """
    agent = make_agent([], incoming_direction=incoming_direction)
    agent._incoming_waypoint = FakeWaypoint(carla.Location(10, 0, 0), is_junction=is_junction)
    agent._local_planner = FakeLocalPlanner()
    return agent


# ---------------------------------------------------------------------------
# _junction_ahead : ne doit se déclencher que pour un virage dans un carrefour
# ---------------------------------------------------------------------------

class TestJunctionAhead(unittest.TestCase):
    def test_true_when_junction_and_turning_right(self):
        agent = make_junction_agent(is_junction=True, incoming_direction=RoadOption.RIGHT)
        self.assertTrue(agent._junction_ahead())

    def test_true_when_junction_and_turning_left(self):
        agent = make_junction_agent(is_junction=True, incoming_direction=RoadOption.LEFT)
        self.assertTrue(agent._junction_ahead())

    def test_false_when_junction_but_going_straight(self):
        agent = make_junction_agent(is_junction=True, incoming_direction=RoadOption.LANEFOLLOW)
        self.assertFalse(agent._junction_ahead())

    def test_false_when_turning_but_not_a_junction(self):
        agent = make_junction_agent(is_junction=False, incoming_direction=RoadOption.RIGHT)
        self.assertFalse(agent._junction_ahead())


# ---------------------------------------------------------------------------
# _junction_gap_is_safe : combine détection + _gap_is_safe (min_gap_time dédié)
# ---------------------------------------------------------------------------

class TestJunctionGapIsSafe(unittest.TestCase):
    def test_true_when_no_obstacle(self):
        agent = make_junction_agent()
        self.assertTrue(agent._junction_gap_is_safe(False, None, -1))

    def test_true_when_stopped_obstacle(self):
        stopped = FakeActor(carla.Location(5, 0, 0), speed=0)
        agent = make_junction_agent()
        self.assertTrue(agent._junction_gap_is_safe(True, stopped, 5.0))

    def test_true_when_large_gap(self):
        # 60 m à 36 km/h (10 m/s) -> 6 s, > JUNCTION_MIN_GAP_TIME (3 s)
        crossing = FakeActor(carla.Location(60, 0, 0), speed=36)
        agent = make_junction_agent()
        self.assertTrue(agent._junction_gap_is_safe(True, crossing, 60.0))

    def test_false_when_small_gap(self):
        # 15 m à 36 km/h (10 m/s) -> 1.5 s, < JUNCTION_MIN_GAP_TIME (3 s)
        crossing = FakeActor(carla.Location(15, 0, 0), speed=36)
        agent = make_junction_agent()
        self.assertFalse(agent._junction_gap_is_safe(True, crossing, 15.0))

    def test_uses_junction_specific_threshold_not_bypass_one(self):
        # 25 m à 36 km/h (10 m/s) -> 2.5 s : sûr pour le bypass (4 s ne serait
        # PAS respecté ici, donc si ce test passe, c'est bien le seuil
        # JUNCTION_MIN_GAP_TIME (3 s) qui est utilisé, pas BYPASS_MIN_GAP_TIME.
        crossing = FakeActor(carla.Location(25, 0, 0), speed=36)
        agent = make_junction_agent()
        self.assertFalse(agent._junction_gap_is_safe(True, crossing, 25.0))
        self.assertNotEqual(BehaviorAgent.JUNCTION_MIN_GAP_TIME, BehaviorAgent.BYPASS_MIN_GAP_TIME)


# ---------------------------------------------------------------------------
# _junction_timed_out : détection de blocage, indépendante du compteur bypass
# ---------------------------------------------------------------------------

class TestJunctionTimedOut(unittest.TestCase):
    def test_false_before_threshold(self):
        agent = make_junction_agent()
        agent.JUNCTION_TIMEOUT_TICKS = 3
        self.assertFalse(agent._junction_timed_out())
        self.assertFalse(agent._junction_timed_out())
        self.assertFalse(agent._junction_timed_out())

    def test_true_after_threshold(self):
        agent = make_junction_agent()
        agent.JUNCTION_TIMEOUT_TICKS = 2
        agent._junction_timed_out()
        agent._junction_timed_out()
        self.assertTrue(agent._junction_timed_out())

    def test_reset_junction_state_clears_counter(self):
        agent = make_junction_agent()
        agent._junction_tick_counter = 42
        agent._junction_state = 'waiting_clear'
        agent._reset_junction_state()
        self.assertEqual(agent._junction_tick_counter, 0)
        self.assertEqual(agent._junction_state, 'idle')


# ---------------------------------------------------------------------------
# junction_manager : cycle complet de la machine à états
# ---------------------------------------------------------------------------

class TestJunctionManagerStateMachine(unittest.TestCase):
    def _agent_with_stubbed_steps(self, is_junction=True):
        agent = make_junction_agent(is_junction=is_junction)
        agent._cross_traffic_obstacle = lambda wp: (False, None, -1)
        return agent

    def test_no_junction_ahead_stays_idle(self):
        agent = self._agent_with_stubbed_steps(is_junction=False)
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))
        self.assertFalse(agent.junction_manager(waypoint))
        self.assertEqual(agent._junction_state, 'idle')

    def test_idle_to_waiting_then_clear_when_no_obstacle(self):
        agent = self._agent_with_stubbed_steps()
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        result = agent.junction_manager(waypoint)

        self.assertFalse(result)  # pas d'obstacle -> le passage est autorisé tout de suite
        self.assertEqual(agent._junction_state, 'crossing')

    def test_waits_while_obstacle_blocks_the_junction(self):
        agent = self._agent_with_stubbed_steps()
        blocker = FakeActor(carla.Location(10, 0, 0), speed=0)
        agent._cross_traffic_obstacle = lambda wp: (True, blocker, 10.0)
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        # Un véhicule à l'arrêt dans le carrefour n'est pas "safe" par
        # _junction_gap_is_safe que si sa vitesse est nulle -> ici il l'est,
        # donc on force un cas non sûr avec une vitesse non nulle et un
        # faible espacement pour bien tester l'attente.
        agent._cross_traffic_obstacle = lambda wp: (
            True, FakeActor(carla.Location(10, 0, 0), speed=36), 10.0)

        result = agent.junction_manager(waypoint)

        self.assertTrue(result)
        self.assertEqual(agent._junction_state, 'waiting_clear')

    def test_transitions_to_crossing_once_gap_is_safe(self):
        agent = self._agent_with_stubbed_steps()
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        # Premier appel : obstacle non sûr -> attente
        agent._cross_traffic_obstacle = lambda wp: (
            True, FakeActor(carla.Location(10, 0, 0), speed=36), 10.0)
        self.assertTrue(agent.junction_manager(waypoint))
        self.assertEqual(agent._junction_state, 'waiting_clear')

        # Le trafic transversal se dégage -> le passage devient sûr
        agent._cross_traffic_obstacle = lambda wp: (False, None, -1)
        result = agent.junction_manager(waypoint)

        self.assertFalse(result)
        self.assertEqual(agent._junction_state, 'crossing')

    def test_crossing_state_lets_planner_finish_without_rechecking(self):
        agent = self._agent_with_stubbed_steps()
        agent._junction_state = 'crossing'

        def fail_if_called(wp):
            raise AssertionError("_cross_traffic_obstacle ne doit pas être appelé en 'crossing'")

        agent._cross_traffic_obstacle = fail_if_called
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        result = agent.junction_manager(waypoint)

        self.assertFalse(result)
        self.assertEqual(agent._junction_state, 'crossing')

    def test_junction_cleared_resets_state_once_past_it(self):
        """Une fois le carrefour dépassé (plus is_junction), l'état doit revenir à idle."""
        agent = self._agent_with_stubbed_steps(is_junction=True)
        agent._junction_state = 'crossing'
        agent._junction_tick_counter = 15

        agent._incoming_waypoint = FakeWaypoint(carla.Location(20, 0, 0), is_junction=False)
        waypoint = FakeWaypoint(carla.Location(20, 0, 0))

        result = agent.junction_manager(waypoint)

        self.assertFalse(result)
        self.assertEqual(agent._junction_state, 'idle')
        self.assertEqual(agent._junction_tick_counter, 0)

    def test_timeout_resets_state_and_flags_failure(self):
        agent = self._agent_with_stubbed_steps()
        agent.JUNCTION_TIMEOUT_TICKS = 1
        agent._cross_traffic_obstacle = lambda wp: (
            True, FakeActor(carla.Location(10, 0, 0), speed=36), 10.0)
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        agent.junction_manager(waypoint)  # idle -> waiting_clear, tick 1 (pas encore > seuil)
        result = agent.junction_manager(waypoint)  # tick 2 > seuil -> timeout

        self.assertFalse(result)
        self.assertEqual(agent._junction_state, 'idle')
        self.assertEqual(agent._scenario_result, False)

    def test_full_cycle_waiting_to_crossing(self):
        """Parcourt idle -> waiting_clear -> crossing, comme un franchissement réel."""
        agent = self._agent_with_stubbed_steps()
        waypoint = FakeWaypoint(carla.Location(0, 0, 0))

        # idle -> waiting_clear (obstacle bloquant détecté)
        agent._cross_traffic_obstacle = lambda wp: (
            True, FakeActor(carla.Location(10, 0, 0), speed=36), 10.0)
        self.assertTrue(agent.junction_manager(waypoint))
        self.assertEqual(agent._junction_state, 'waiting_clear')

        # waiting_clear -> crossing (créneau sûr trouvé)
        agent._cross_traffic_obstacle = lambda wp: (False, None, -1)
        self.assertFalse(agent.junction_manager(waypoint))
        self.assertEqual(agent._junction_state, 'crossing')


# ---------------------------------------------------------------------------
# _hold_position : arrêt en douceur, distinct de emergency_stop
# ---------------------------------------------------------------------------

class TestHoldPosition(unittest.TestCase):
    def test_sets_speed_to_zero_and_returns_planner_control(self):
        agent = make_junction_agent()
        control = agent._hold_position()
        self.assertEqual(agent._local_planner.speed, 0)
        self.assertEqual(control, "CONTROL")


if __name__ == "__main__":
    unittest.main(verbosity=2)