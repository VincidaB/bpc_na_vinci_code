## Install FoudationPose in conda environment

(instructions based on FoudantionPose's readme and issues)


In the root of the FoundationPose repository:
```bash
conda create -n foundationpose python=3.9
conda activate foundationpose

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -c conda-forge gcc=11 gxx=11
conda install -c conda-forge boost-cpp


conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX/include/eigen3"

python -m pip install -r requirements.txt
python -m pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```
Make sure to run `build_all_conda.sh` and not `build_all.sh` as the latter will not work in a conda environment.
