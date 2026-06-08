docker run --rm \
        -v $(pwd)/userCode/:/workspace/team_code/ \
        --network=bridge \
        --name carla-client-instance-${USER} \
        -p 8833:8833\
        -p 9833:9833 \
        -d\
        -it carla-client \
        /bin/bash
