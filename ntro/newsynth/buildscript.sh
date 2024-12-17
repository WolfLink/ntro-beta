docker build -t gridsynth .
docker run --name tmp-gridsynth gridsynth
docker cp tmp-gridsynth:/root/gridsynth ./gridsynth
docker rm -f tmp-gridsynth
