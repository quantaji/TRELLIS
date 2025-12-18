# Read Arguments
TEMP=$(getopt -o h --long help,new-env,basic,train,xformers,flash-attn,diffoctreerast,vox2seq,spconv,mipgaussian,kaolin,nvdiffrast,demo -n 'setup.sh' -- "$@")

eval set -- "$TEMP"

HELP=false
NEW_ENV=true
BASIC=true
TRAIN=false
XFORMERS=true
FLASHATTN=true
DIFFOCTREERAST=true
VOX2SEQ=false
LINEAR_ASSIGNMENT=false
SPCONV=true
ERROR=false
MIPGAUSSIAN=true
KAOLIN=true
NVDIFFRAST=true
DEMO=false

if [ "$#" -eq 1 ]; then
    HELP=true
fi

while true; do
    case "$1" in
    -h | --help)
        HELP=true
        shift
        ;;
    --new-env)
        NEW_ENV=true
        shift
        ;;
    --basic)
        BASIC=true
        shift
        ;;
    --train)
        TRAIN=true
        shift
        ;;
    --xformers)
        XFORMERS=true
        shift
        ;;
    --flash-attn)
        FLASHATTN=true
        shift
        ;;
    --diffoctreerast)
        DIFFOCTREERAST=true
        shift
        ;;
    --vox2seq)
        VOX2SEQ=true
        shift
        ;;
    --spconv)
        SPCONV=true
        shift
        ;;
    --mipgaussian)
        MIPGAUSSIAN=true
        shift
        ;;
    --kaolin)
        KAOLIN=true
        shift
        ;;
    --nvdiffrast)
        NVDIFFRAST=true
        shift
        ;;
    --demo)
        DEMO=true
        shift
        ;;
    --)
        shift
        break
        ;;
    *)
        ERROR=true
        break
        ;;
    esac
done

if [ "$ERROR" = true ]; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ]; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new conda environment"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --xformers              Install xformers"
    echo "  --flash-attn            Install flash-attn"
    echo "  --diffoctreerast        Install diffoctreerast"
    echo "  --vox2seq               Install vox2seq"
    echo "  --spconv                Install spconv"
    echo "  --mipgaussian           Install mip-splatting"
    echo "  --kaolin                Install kaolin"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --demo                  Install all dependencies for demo"
    return
fi

set -e

echo NEW_ENV=${NEW_ENV}
echo BASIC=${BASIC}
echo TRAIN=${TRAIN}
echo XFORMERS=${XFORMERS}
echo FLASHATTN=${FLASHATTN}
echo DIFFOCTREERAST=${DIFFOCTREERAST}
echo VOX2SEQ=${VOX2SEQ}
echo LINEAR_ASSIGNMENT=${LINEAR_ASSIGNMENT}
echo SPCONV=${SPCONV}
echo ERROR=${ERROR}
echo MIPGAUSSIAN=${MIPGAUSSIAN}
echo KAOLIN=${KAOLIN}
echo NVDIFFRAST=${NVDIFFRAST}
echo DEMO=${DEMO}

export INSTALLED_PYTHON_VERSION="3.10"
export INSTALLED_CUDA_VERSION="11.8.0"
export INSTALLED_CUDA_ABBREV="cu118"
export INSTALLED_GCC_VERSION="11.2.0"
export INSTALLED_PYTORCH_VERSION="2.4.0"
export INSTALLED_TORCHVISION_VERSION="0.19.0"
env_name=trellis

if [ "$NEW_ENV" = true ]; then
    conda create --name $env_name --yes python=${INSTALLED_PYTHON_VERSION}
fi

eval "$(conda shell.bash hook)"
conda activate $env_name
conda install -y -c conda-forge gxx=${INSTALLED_GCC_VERSION}
conda install -y -c "nvidia/label/cuda-${INSTALLED_CUDA_VERSION}" cuda
pip install torch==${INSTALLED_PYTORCH_VERSION} torchvision==${INSTALLED_TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/${INSTALLED_CUDA_ABBREV}

# Get system information
WORKDIR=$(pwd)
# PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PYTORCH_VERSION="2.4.0"
# PLATFORM=$(python -c "import torch; print(('cuda' if torch.version.cuda else ('hip' if torch.version.hip else 'unknown')) if torch.cuda.is_available() else 'cpu')")
PLATFORM="cuda"

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export AM_I_DOCKER=1
export BUILD_WITH_CUDA=1
export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export FORCE_CUDA=1
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.7;8.9;9.0+PTX"
export PYTHONPATH=${WORKDIR}:${PYTHONPATH:-}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

unset CFLAGS CXXFLAGS
export CFLAGS="-O3 -march=x86-64 -mtune=generic -frecord-gcc-switches"
export CXXFLAGS="$CFLAGS"
export CMAKE_VERBOSE_MAKEFILE=1
export VERBOSE=1
export NVCC_FLAGS="$NVCC_FLAGS -Xcompiler -frecord-gcc-switches"

case $PLATFORM in
cuda)
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
    CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)
    echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
    ;;
hip)
    HIP_VERSION=$(python -c "import torch; print(torch.version.hip)")
    HIP_MAJOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f1)
    HIP_MINOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f2)
    # Install pytorch 2.4.1 for hip
    if [ "$PYTORCH_VERSION" != "2.4.1+rocm6.1" ]; then
        echo "[SYSTEM] Installing PyTorch 2.4.1 for HIP ($PYTORCH_VERSION -> 2.4.1+rocm6.1)"
        pip install -vvv torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/rocm6.1 --user
        mkdir -p /tmp/extensions
        sudo cp /opt/rocm/share/amd_smi /tmp/extensions/amd_smi -r
        cd /tmp/extensions/amd_smi
        sudo chmod -R 777 .
        pip install -vvv .
        cd $WORKDIR
        PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    fi
    echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, HIP Version: $HIP_VERSION"
    ;;
*) ;;
esac

if [ "$BASIC" = true ]; then
    pip install -vvv pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers pandas objaverse huggingface_hub open_clip_torch black ipykernel
    pip install -vvv git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
fi

if [ "$TRAIN" = true ]; then
    pip install -vvv tensorboard pandas lpips
    conda install -y -c conda-forge zlib libjpeg-turbo freetype lcms2 libtiff libwebp openjpeg
    pip uninstall -y pillow
    pip install -vvv pillow-simd
fi

if [ "$XFORMERS" = true ]; then
    # install xformers
    if [ "$PLATFORM" = "cuda" ]; then
        if [ "$CUDA_VERSION" = "11.8" ]; then
            case $PYTORCH_VERSION in
            2.0.1) pip install -vvv https://files.pythonhosted.org/packages/52/ca/82aeee5dcc24a3429ff5de65cc58ae9695f90f49fbba71755e7fab69a706/xformers-0.0.22-cp310-cp310-manylinux2014_x86_64.whl ;;
            2.1.0) pip install -vvv xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.1.1) pip install -vvv xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.1.2) pip install -vvv xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.2.0) pip install -vvv xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.2.1) pip install -vvv xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.2.2) pip install -vvv xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.3.0) pip install -vvv xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.4.0) pip install -vvv xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.4.1) pip install -vvv xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118 ;;
            2.5.0) pip install -vvv xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118 ;;
            *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.1" ]; then
            case $PYTORCH_VERSION in
            2.1.0) pip install -vvv xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.1.1) pip install -vvv xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.1.2) pip install -vvv xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.2.0) pip install -vvv xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.2.1) pip install -vvv xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.2.2) pip install -vvv xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.3.0) pip install -vvv xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.4.0) pip install -vvv xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.4.1) pip install -vvv xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121 ;;
            2.5.0) pip install -vvv xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121 ;;
            *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        elif [ "$CUDA_VERSION" = "12.4" ]; then
            case $PYTORCH_VERSION in
            2.5.0) pip install -vvv xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124 ;;
            *) echo "[XFORMERS] Unsupported PyTorch & CUDA version: $PYTORCH_VERSION & $CUDA_VERSION" ;;
            esac
        else
            echo "[XFORMERS] Unsupported CUDA version: $CUDA_MAJOR_VERSION"
        fi
    elif [ "$PLATFORM" = "hip" ]; then
        case $PYTORCH_VERSION in
        2.4.1\+rocm6.1) pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/rocm6.1 ;;
        *) echo "[XFORMERS] Unsupported PyTorch version: $PYTORCH_VERSION" ;;
        esac
    else
        echo "[XFORMERS] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$FLASHATTN" = true ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        pip install -vvv --no-build-isolation flash-attn
    elif [ "$PLATFORM" = "hip" ]; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.6.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$KAOLIN" = true ]; then
    # install kaolin
    if [ "$PLATFORM" = "cuda" ]; then
        case $PYTORCH_VERSION in
        2.0.1) pip install -vvv kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html ;;
        2.1.0) pip install -vvv kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html ;;
        2.1.1) pip install -vvv kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html ;;
        2.2.0) pip install -vvv kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.0_cu118.html ;;
        2.2.1) pip install -vvv kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.1_cu118.html ;;
        2.2.2) pip install -vvv kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu118.html ;;
        2.4.0) pip install -vvv kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html ;;
        *) echo "[KAOLIN] Unsupported PyTorch version: $PYTORCH_VERSION" ;;
        esac
    else
        echo "[KAOLIN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        mkdir -p /tmp/extensions
        git clone --branch v0.3.5 --depth 1 --recurse-submodules https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        pip install -vvv --no-build-isolation /tmp/extensions/nvdiffrast
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$DIFFOCTREERAST" = true ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        mkdir -p /tmp/extensions
        git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
        pip install -vvv --no-build-isolation /tmp/extensions/diffoctreerast
    else
        echo "[DIFFOCTREERAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$MIPGAUSSIAN" = true ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        mkdir -p /tmp/extensions
        git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
        pip install -vvv --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
    else
        echo "[MIPGAUSSIAN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$VOX2SEQ" = true ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        mkdir -p /tmp/extensions
        cp -r extensions/vox2seq /tmp/extensions/vox2seq
        pip install -vvv --no-build-isolation /tmp/extensions/vox2seq
    else
        echo "[VOX2SEQ] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$SPCONV" = true ]; then
    # install spconv
    if [ "$PLATFORM" = "cuda" ]; then
        case $CUDA_MAJOR_VERSION in
        11) pip install -vvv spconv-cu118 ;;
        12) pip install -vvv spconv-cu120 ;;
        *) echo "[SPCONV] Unsupported PyTorch CUDA version: $CUDA_MAJOR_VERSION" ;;
        esac
    else
        echo "[SPCONV] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$DEMO" = true ]; then
    pip install -vvv gradio==4.44.1 gradio_litmodel3d==0.0.1
fi
