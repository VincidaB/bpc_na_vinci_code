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


To run with the demo data in the repo just run : 
```bash
python demo_ipd.py --debug 0 --est_refine_iter 3
```
Note :
might need to run this in every terminal session in the conda environment to set the library path for the conda environment:
```bash 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
Or do the following to set it permanently:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

echo '#!/bin/sh
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/set_env.sh

echo '#!/bin/sh
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/unset_env.sh

chmod +x $CONDA_PREFIX/etc/conda/activate.d/set_env.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/unset_env.sh
```

Or if using fish shell :
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo '#!/usr/bin/fish
set -gx OLD_LD_LIBRARY_PATH $LD_LIBRARY_PATH
set -gx LD_LIBRARY_PATH $CONDA_PREFIX/lib $LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/set_env.fish

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo '#!/usr/bin/fish
set -gx LD_LIBRARY_PATH $OLD_LD_LIBRARY_PATH
set -e OLD_LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/unset_env.fish

chmod +x $CONDA_PREFIX/etc/conda/activate.d/set_env.fish
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/unset_env.fish
```

