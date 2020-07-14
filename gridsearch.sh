#!/usr/bin/env bash

rsync -rtuv /Users/benediktvogler/Studium/Masterarbeit/SNNexperiments/* ubuntu:~/Dokumente/SNNexperiments/
ssh ubuntu << EOF
cd ~/Dokumente/SNNexperiments/
source /home/bsvogler/Dokumente/nest/bin/nest_vars.sh
screen -S "Gridsearch" -d -m
screen -r "Gridsearch" -X stuff $'python3 ./experiments/polebalancing.py --processes=6 -g gridsearch.json --headless;rsync -rtuv ./experimentdata benediktvogler@192.168.2.3:~/Studium/Masterarbeit/SNNexperiments/;\n'
EOF