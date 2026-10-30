#!/usr/bin/env bash
# Build the AeroTrack training environment on the HPC (RHEL 7.9, glibc 2.17, driver 510.47 = CUDA 11.6, Tesla V100 sm_70).
# Constraints: wheels must be manylinux2014 (glibc 2.17); CUDA runtime 11.8 runs on driver 510 via CUDA minor-version
# compatibility (sm_70 SASS is in the wheels, so no PTX JIT is needed). The OMEN runs torch 2.13+cu130 -- a cu13 build
# cannot run on driver 510, so the HPC torch differs from the OMEN by necessity (recorded in docs/REMOTE_STATE.md).
# Usage (login node, has internet): nohup scripts/hpc/build_env.sh <env_prefix> > log 2>&1 &
set -eu
ENV="$1"
source ~/anaconda3/etc/profile.d/conda.sh
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.11
conda activate "$ENV"
python -m pip install --upgrade pip
ok=""
for v in "2.7.1 0.22.1" "2.6.0 0.21.0" "2.5.1 0.20.1"; do
  set -- $v
  if python -m pip install --only-binary=:all: "torch==$1" "torchvision==$2" --index-url https://download.pytorch.org/whl/cu118; then ok=$1; break; fi
done
[ -n "$ok" ] || { echo "NO_TORCH_WHEEL"; exit 2; }
python -m pip install --only-binary=:all: "numpy>=2.1,<3" "pandas>=2.2" "polars>=1.20" "scipy>=1.14" "matplotlib>=3.9" "pillow>=11" \
  "opencv-python-headless>=4.10" "PyYAML>=6.0" "tqdm>=4.66" "psutil>=6.0" "requests>=2.32" "nvidia-ml-py>=12.560" \
  "ultralytics==8.4.115" "ultralytics-thop>=2.0" pyarrow onnx onnxruntime
# ultralytics pulls opencv-python (GUI build); keep the headless one only if both resolve -- record what is installed
python -m pip freeze > "$ENV/../$(basename $ENV)_freeze.txt"
python - <<'PY'
import torch, torchvision, numpy, cv2, ultralytics, platform
print("ENV_OK", platform.python_version(), "torch", torch.__version__, "cuda_build", torch.version.cuda,
      "tv", torchvision.__version__, "numpy", numpy.__version__, "cv2", cv2.__version__, "ultralytics", ultralytics.__version__)
PY
