# pointnet-cpp
Pytorch cpp implementation of Pointnet
```bash
#clone
git clone https://github.com/harishkool/pointnet-cpp.git 

#get libtorch, this is for cuda-10.0
#you can install pytorch from source as well.
wget https://download.pytorch.org/libtorch/cu100/libtorch-cxx11-abi-shared-with-deps-1.3.0.zip

#install high five for loading .h5 files
https://github.com/BlueBrain/HighFive

#dataset
train.h5 and test.h5 files are provided along with the repo.

#build instructions
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="path_to/libtorch" ..
make
./pointnet
```
### Using Pytorch Dataloader ###
Right now there are some issues with integrating compiled libtorch and pcl or open3d related to pytorch.
Once it is resolved, will update the dataloader by integrating pcl or open3d.
