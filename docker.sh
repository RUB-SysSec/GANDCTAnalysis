#!/bin/bash
DATASET_DIRS="$HOME/datasets"

build()
{
    docker build -t dct .
}

shell() 
{
    docker run --gpus all -it -v $DATASET_DIRS:/dct/datasets -v $(pwd):/dct dct
}

tests() 
{
    docker run -it -v $(pwd):/dct dct
}

clean() 
{
    rm -r log ckpt final_models
}


print_usage() 
{
    echo "Choose: docker.sh {build|download|convert|shell}"
    echo "    build - Build the Dockerfile."
    echo "    shell - Spawn a shell inside the docker container."
    echo "    tests - Spawn Docker instance for pytest."
    echo "    clean - Cleanup directories from training."
}

if [[ $1 == "" ]]; then
    echo "No argument provided"
    print_usage
elif [[ $1 == "build" ]]; then
    build
elif [[ $1 == "shell" ]]; then
    shell 
elif [[ $1 == "tests" ]]; then
    shell 
elif [[ $1 == "clean" ]]; then
    clean 
else 
    echo "Argument not recognized!."
    print_usage
fi

