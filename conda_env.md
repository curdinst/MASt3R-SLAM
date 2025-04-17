# Install on asl-server







conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit 
cd MASt3R-SLAM/
pip install faiss-cpu==1.7.4
export CC=gcc-9
export CXX=g++-9
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .


# gaussian optimisation steps:
cd thirdparty/
git submodule add https://github.com/rmurai061^Cdiff-gaussian-rasterization-w-pose.git
git submodule add https://gitlab.inria.fr/bkerbl/simple-knn.git
cd ..

conda install gxx_linux-64
pip install thirdparty/simple-knn/
y
conda install conda-forge::glm
y
pip install thirdparty/diff-gaussian-rasterization-w-pose/
y

pip install munch

pip install open3d

<!-- conda install pytorch3d -->
