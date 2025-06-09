#!/usr/bin/zsh
#SBATCH --job-name=unitcell
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-24:00:00
#SBATCH --mem=16G
#SBATCH --output=unitcell_%j.out    # Output file (stdout)
#SBATCH --error=unitcell_%j.err     # Error file (stderr)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=ulumuddin@imm.rwth-aachen.de


export TF_CPP_MIN_LOG_LEVEL=0
export CUDA_VISIBLE_DEVICES=0
export TF_ENABLE_ONEDNN_OPTS=0

export NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))
for dir in ${NVIDIA_DIR}/*; do if [ -d "${dir}/lib" ]; then export LD_LIBRARY_PATH="${dir}/lib:${LD_LIBRARY_PATH}"; fi; done
export PATH=${NVIDIA_DIR}/cuda_nvcc/bin:${PATH}

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}




module purge
# build 15

module load GCC/12.3.0
module load intel-compilers/2022.1.0
module load OpenSSL/1.1
module load GCCcore/.12.3.0
module load zlib/1.2.13
module load binutils/2.40
module load numactl/2.0.16
module load NVHPC/24.3-CUDA-12.3.0
module load libxml2/2.11.4
module load libpciaccess/0.17
module load hwloc/2.9.1
module load PMIx/4.2.4
module load UCC/1.2.0
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load nvompi/2023.07
module load libevent/2.1.12
module load XZ/5.4.2
module load libarchive/3.6.2
module load GMP/6.2.1
module load libffi/3.4.2
module load ncurses/6.4
module load FFTW/3.3.10
module load imkl/2021.4.0
module load Eigen/3.4.0
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
module load UCX/1.15.0
module load GDRCopy/2.4
module load NCCL/2.20.5-CUDA-12.3.0
module load Z3/4.12.2
module load NASM/2.16.01
module load bzip2/1.0.8
module load x264/20230226
module load LAME/3.100
module load x265/3.5
module load expat/2.5.0
module load libpng/1.6.39
module load Brotli/1.0.9
module load freetype/2.13.0
module load util-linux/2.39
module load fontconfig/2.14.2
module load xorg-macros/1.20.0
module load X11/20230603
module load FriBidi/1.0.12
module load SDL2/2.28.2
module load FFmpeg/6.0
module load UCX-CUDA/1.15.0-CUDA-12.3.0
module load Clang/18.1.2-CUDA-12.3.0


source ~/miniconda3/etc/profile.d/conda.sh
conda activate grace



ulimit -u 260000


# Run the simulation
srun  /home/wz300646/Installations/lammps_v02/lammps/build/lmp -in fire_cg.in


