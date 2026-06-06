docker run --rm --gpus '"device=DEV"' --net=bridge -p CP-P2:CP-P2 --name carla_server_${USER} -d carla_leaderboard_2.0:latest /bin/bash ./CarlaUE4.sh -carla-port=CP -RenderOffScreen
