# ssc_exploration


mkdir -p ssc_ws/src
cd ssc_ws/src
git clone --recurse-submodules https://github.com/arnovaill/ssc_exploration.git
git clone https://github.com/eric-wieser/ros_numpy.git
cd ..

wstool init
wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
wstool up
rosdep install --from-paths src --ignore-src -y -r

catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE='/home/arno/anaconda3/envs/ssc/bin/python3.6' -DPYTHON_INCLUDE_DIR='/home/arno/anaconda3/envs/ssc/include/python3.6m' -DPYTHON_LIBRARY='/home/arno/anaconda3/envs/ssc/lib/libpython3.6m.so'

