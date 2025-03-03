#!/usr/bin/env bash
# ref: https://gist.github.com/jehiah/855086

function usage() {
    echo "build"
    echo ""
    echo "./build.sh"
    echo "    -h --help"
    echo "    --push-maze-base push maze-base container image"
    echo ""
}

function push_maze_base() {
    local maze_base_tag=0.0.1
    docker build \
      -t 022497628448.dkr.ecr.us-west-2.amazonaws.com/pet-projects/maze-base:${maze_base_tag} \
      -f docker/maze-base/Dockerfile . && \
    docker push 022497628448.dkr.ecr.us-west-2.amazonaws.com/pet-projects/maze-base:${maze_base_tag}
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --push-maze-base)
            push_maze_base
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done
