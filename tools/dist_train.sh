#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
# PORT=${PORT:-28509}
# PORT=${PORT:-28512}
# PORT=${PORT:-28519}
# PORT=${PORT:-28523}
# PORT=${PORT:-28521}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
