docker run --rm --gpus '"device=0"' --net=bridge -p 6034:6034 --name carla_server_${USER} -d carla_leaderboard_2.0:latest /bin/bash ./CarlaUE4.sh -carla-port=6033 -RenderOffScreen
