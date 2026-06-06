docker run --rm \
        -v $(pwd)/userCode/:/workspace/team_code/ \
        --network=bridge \
        --name carla-client-instance-${USER} \
        -p TP:TP\
        -p VP:VP \
        -d\
        -it carla-client \
        /bin/bash
