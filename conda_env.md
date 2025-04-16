# Install on asl-server

conda create -n mast3r-slam python=3.11
conda activate mast3r-slamconda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install nvidia/label/cuda-11.8.0::cuda-toolkitgit clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/pip install faiss-cpu==1.7.4export CC=gcc-9
export CXX=g++-9pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .