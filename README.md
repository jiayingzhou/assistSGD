# assistSGD

This file contains experiment code on Deep Neural Network and Reinforcement Learning.

## assistRL.py
This is the code about using assistSGD on Reinforcement Learning senario. 
To run the code, you need to use conda to install some packages, including:

conda install -c conda-forge gym

conda install -c conda-forge gym-recording

conda install -c pytorch pytorch

conda install -c conda-forge box2d-py

conda install -c conda-forge ffmpeg

conda install -c conda-forge pyglet

conda install tensorboard

conda install -c conda-forge asciinema

conda install -c conda-forge gym=0.17.3 (If the video cannot be played, try this package)

### Lunarlander_V2.py
To run the code on the customized environment Lunarlander_V2.py, please find where your gym locates, go to gym/envs/box2d/ paste the Lunarlander_V2.py under this file.

After installing packages and paste the file, if you don't want record, run the code python3 assistRL.py; If you want to get recorded videos, run python3 assistRL.py --record=True

