#!/bin/bash
#qdtrack_ training.xml
# export ROUTES=/workspace/team_code/route_controlling.xml
export ROUTES=/workspace/team_code/route_1_avddiem.xml
#export ROUTES=${LEADERBOARD_ROOT}/data/routes_devtest.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=/workspace/team_code/carla_behavior_agent/basic_autonomous_agent.py
export TEAM_CONFIG=/workspace/team_code/carla_behavior_agent/config_agent_basic.json
export CHALLENGE_TRACK_CODENAME=SENSORS
export CARLA_HOST=172.16.174.233
export CARLA_PORT=6033
export CARLA_TRAFFIC_MANAGER_PORT=8833
export CHECKPOINT_ENDPOINT=/workspace/team_code/simulation_results_r1.json
export DEBUG_CHECKPOINT_ENDPOINT=/workspace/team_code/results/live_results.txt
export RESUME=false
export TIMEOUT=60

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--host=${CARLA_HOST} \
--port=${CARLA_PORT} \
--timeout=${TIMEOUT} \
--traffic-manager-port=${CARLA_TRAFFIC_MANAGER_PORT} 
