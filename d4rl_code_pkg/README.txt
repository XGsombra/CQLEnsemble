This zip file contains code for the paper D4RL: Datasets for Deep Data-Driven Reinforcement Learning.

There are two subdirectories:
- d4rl contains the environment code. This can be installed by simply running `pip install -e .` inside the subdirectory.
- evaluation_code contains the algorithm implementations used for benchmarking on the dataset. Each algorithm has a python
  script which will run the appropriate experiment.

In order to download the datasets, you need to install gsutil (https://cloud.google.com/storage/docs/gsutil_install?hl=pl).
Simply creating the environment with gym.make(env_name) and then calling env.get_dataset() will download the dataset automatically.

UPDATE
--
For the new Bullet datasets, dataset URLs are supplied in the
d4rl/bullet_datasets.txt folder, and can be loaded as an hdf5 file in the same
format as the existing datasets. The environments can be imported with the
ids:
bullet-halfcheetah-v0
bullet-hopper-v0
bullet-ant-v0
bullet-walker2d-v0
bullet-maze2d-open-v0
bullet-maze2d-umaze-v0
bullet-maze2d-medium-v0
bullet-maze2d-large-v0
