# Sourced by every HPC job (tools/hpc/run_job.pbs) and by interactive checks on the HPC:
#   source ~/aerotrack_server/source/tools/hpc/job_env.sh [staged]
# Activates the pinned environment and sets the variables that make the OMEN-written manifests usable on Linux.
#   AEROTRACK_PATH_MAP  OMEN prefix -> the data root (node-local /local copy when "staged", else the NFS copy)
#   AEROTRACK_CORPUS    corpus v2 (all new work)
#   AEROTRACK_AMP       fp16 (V100: no native bf16; D-2026-10-06-04)
export AT_HOME="$HOME/aerotrack_server"
export AT_SRC="$AT_HOME/source"
export AT_ENV="$AT_HOME/environments/at311"
export AT_NFS_ROOT="$AT_HOME/shared_data/root"
export AT_LOCAL_ROOT="/local/$USER/at_root"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$AT_ENV"
if [ "${1:-}" = "staged" ]; then AT_DATA_ROOT="$AT_LOCAL_ROOT"; else AT_DATA_ROOT="$AT_NFS_ROOT"; fi
export AT_DATA_ROOT
export AEROTRACK_PATH_MAP='A:\AEROTRACK_OMEN=>'"$AT_DATA_ROOT"
export AEROTRACK_CORPUS="${AEROTRACK_CORPUS:-data_unified_v2}"
export AEROTRACK_AMP="${AEROTRACK_AMP:-fp16}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$AT_SRC"
