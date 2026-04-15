# Read Arguments
TEMP=`getopt -o h --long help,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false


if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --nvdiffrec) NVDIFFREC=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --basic                 Install basic dependencies"
    echo "  --flash-attn            Install flash-attention"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --nvdiffrast            Install nvdiffrast (CUDA) / nvdiffrast-hip (ROCm)"
    echo "  --nvdiffrec             Install nvdiffrec (CUDA only)"
    echo ""
    echo "  Activate your Python environment before running this script."
    echo "  For ROCm, ensure ROCm PyTorch is installed first:"
    echo "    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4"
    echo "  For CUDA:"
    echo "    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124"
    return
fi

# Get system information
WORKDIR=$(pwd)
if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU found"
    exit 1
fi


if [ "$BASIC" = true ] ; then
    pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh "transformers==4.56.0" gradio==6.0.1 tensorboard pandas lpips zstandard pyfqmr matplotlib
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    sudo apt install -y libjpeg-dev
    pip install pillow-simd
    pip install kornia timm
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        pip install flash-attn==2.7.3
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx1201 python setup.py install
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        pip install /tmp/extensions/nvdiffrast --no-build-isolation
    elif [ "$PLATFORM" = "hip" ] ; then
        mkdir -p /tmp/extensions
        git clone https://github.com/Cardboard-box-a/nvdiffrast-hip.git /tmp/extensions/nvdiffrast-hip
        pip install /tmp/extensions/nvdiffrast-hip --no-build-isolation
    fi
fi

if [ "$NVDIFFREC" = true ] ; then
    mkdir -p /tmp/extensions
    git clone -b renderutils https://github.com/Cardboard-box-a/nvdiffrec.git /tmp/extensions/nvdiffrec
    pip install /tmp/extensions/nvdiffrec --no-build-isolation
fi

if [ "$CUMESH" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/Cardboard-box-a/CuMesh.git /tmp/extensions/CuMesh --recursive
    pip install /tmp/extensions/CuMesh --no-build-isolation
fi

if [ "$FLEXGEMM" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/Cardboard-box-a/FlexGEMM-rocm.git /tmp/extensions/FlexGEMM
    pip install /tmp/extensions/FlexGEMM --no-build-isolation
fi

if [ "$OVOXEL" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/Cardboard-box-a/o-voxel.git /tmp/extensions/o-voxel --recursive
    pip install /tmp/extensions/o-voxel --no-build-isolation
fi
