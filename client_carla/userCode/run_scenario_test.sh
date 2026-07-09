#!/bin/bash
# =============================================================================
# run_scenario_test.sh
# Runs a single test route (1 scenario) by specifying its index.
#
# Usage:
#   ./run_scenario_test.sh          -> interactive menu
#   ./run_scenario_test.sh 1        -> runs index 1 directly (InvadingTurn)
#   ./run_scenario_test.sh 1 -w hard  -> same as above + weather preset (see --weather)
#
# Individual routes are read from the $ROUTE_DIR folder
# (NN_<Type>.xml files, each containing a route with id=0).
# =============================================================================

set -euo pipefail

# --- Individual route file  --------------------------------------------------
ROUTE_DIR="${ROUTE_DIR:-/workspace/team_code/route_per_scenario}"

# --- CARLA settings / agent --------------------------------------------------
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=/workspace/team_code/carla_behavior_agent/basic_autonomous_agent.py
export TEAM_CONFIG=/workspace/team_code/carla_behavior_agent/config_agent_basic.json
export CHALLENGE_TRACK_CODENAME=SENSORS
export CARLA_HOST=172.16.174.233
export CARLA_PORT=6033
export CARLA_TRAFFIC_MANAGER_PORT=8833
export RESUME=false
export TIMEOUT=60
export RECORD_PATH="${RECORD_PATH:-}"

# =============================================================================
# 1) Compile a list of the scenarios available in the folder
# =============================================================================
if [[ ! -d "$ROUTE_DIR" ]]; then
    echo "ERROR : directory not found: $ROUTE_DIR" >&2
    echo "        (Set it via the ROUTE_DIR variable or edit this script.)" >&2
    exit 1
fi

mapfile -t FILES < <(find "$ROUTE_DIR" -maxdepth 1 -type f -name '[0-9][0-9]_*.xml' | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "ERROR: No NN_<Type>.xml file found in $ROUTE_DIR" >&2
    exit 1
fi

declare -A ROUTE_OF        
declare -A TYPE_OF         
for f in "${FILES[@]}"; do
    base="$(basename "$f")"          
    idx="$((10#${base%%_*}))"         
    typ="${base#*_}"; typ="${typ%.xml}"
    ROUTE_OF[$idx]="$f"
    TYPE_OF[$idx]="$typ"
done

print_menu() {
    echo "==================================================================="
    echo " Available test routes (directory: $ROUTE_DIR)"
    echo "==================================================================="
    for idx in $(printf '%s\n' "${!ROUTE_OF[@]}" | sort -n); do
        printf "  [%2d]  %s\n" "$idx" "${TYPE_OF[$idx]}"
    done
    echo "-------------------------------------------------------------------"
}

# =============================================================================
# 2) Resolve the index: argument $1, otherwise an interactive menu
# =============================================================================
WEATHER_PRESET="default"

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--weather) WEATHER_PRESET="${2:-default}"; shift 2 ;;
        -h|--help)
            echo "Usage : $0 [index] [-w default|clear|hard]"; print_menu; exit 0 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]:-}"

INDEX="${1:-}"
if [[ -z "$INDEX" ]]; then
    print_menu
    read -rp "Index of the scenario to be tested : " INDEX
fi

# Validation
if ! [[ "$INDEX" =~ ^[0-9]+$ ]] || [[ -z "${ROUTE_OF[$INDEX]:-}" ]]; then
    echo "ERROR : invalid index -> '$INDEX'" >&2
    print_menu
    exit 1
fi

ROUTE_FILE="${ROUTE_OF[$INDEX]}"
SCN_TYPE="${TYPE_OF[$INDEX]}"

# =============================================================================
# 3) Optional weather preset 
# =============================================================================
RUN_ROUTE="$ROUTE_FILE"
if [[ "$WEATHER_PRESET" != "default" ]]; then
    case "$WEATHER_PRESET" in
        clear) export W_ATTRS='cloudiness="10.0" precipitation="0.0" precipitation_deposits="0.0" wetness="0.0" wind_intensity="10.0" sun_azimuth_angle="-1.0" sun_altitude_angle="45.0" fog_density="0.0"' ;;
        hard)  export W_ATTRS='cloudiness="100.0" precipitation="100.0" precipitation_deposits="90.0" wetness="100.0" wind_intensity="100.0" sun_azimuth_angle="-1.0" sun_altitude_angle="-90.0" fog_density="3.0"' ;;
        *) echo "ERROR : unknown weather preset '$WEATHER_PRESET' (default|clear|hard)" >&2; exit 1 ;;
    esac
    RUN_ROUTE="$(mktemp --suffix=.xml)"
    trap 'rm -f "$RUN_ROUTE"' EXIT
    export SRC_ROUTE="$ROUTE_FILE" DST_ROUTE="$RUN_ROUTE"
    python3 - <<'PY'
import os, re
src, dst, attrs = os.environ["SRC_ROUTE"], os.environ["DST_ROUTE"], os.environ["W_ATTRS"]
txt = open(src, encoding="utf-8").read()
new = re.sub(r'<weather\s+route_percentage="(\d+)"[^>]*?/>',
             lambda m: '<weather route_percentage="%s"\n\t\t%s/>' % (m.group(1), attrs),
             txt)
open(dst, "w", encoding="utf-8").write(new)
PY
    echo ">> Forced weather : $WEATHER_PRESET"
fi

# =============================================================================
# 4) Outings specifically for this scenario 
# =============================================================================
TAG="$(printf '%02d_%s' "$INDEX" "$SCN_TYPE")"
export ROUTES="$RUN_ROUTE"
export ROUTES_SUBSET=0     
export CHECKPOINT_ENDPOINT="/workspace/team_code/results/simulation_results_${TAG}.json"
export DEBUG_CHECKPOINT_ENDPOINT="/workspace/team_code/results/live_results_${TAG}.txt"
mkdir -p "$(dirname "$CHECKPOINT_ENDPOINT")"

echo "==================================================================="
echo " Scenario  : [$INDEX] $SCN_TYPE"
echo " Route     : $ROUTE_FILE"
echo " Results   : $CHECKPOINT_ENDPOINT"
echo "==================================================================="

# =============================================================================
# 5) Launch of the leaderboard
# =============================================================================
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