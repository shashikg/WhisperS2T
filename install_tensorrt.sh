echo "----------------[ Installing OpenMPI ]----------------"
apt-get update && apt-get -y install openmpi-bin libopenmpi-dev || sudo apt-get update && sudo apt-get -y install openmpi-bin libopenmpi-dev

echo "----------------[ Installing MPI4PY ]----------------"
MPI4PY_VERSION="3.1.5"
RELEASE_URL="https://github.com/mpi4py/mpi4py/archive/refs/tags/${MPI4PY_VERSION}.tar.gz"
curl -L ${RELEASE_URL} | tar -zx -C /tmp
# Bypassing compatibility issues with higher versions (>= 69) of setuptools.
sed -i 's/>= 40\.9\.0/>= 40.9.0, < 69/g' /tmp/mpi4py-${MPI4PY_VERSION}/pyproject.toml
pip3 install /tmp/mpi4py-${MPI4PY_VERSION}
rm -rf /tmp/mpi4py*

echo "----------------[ Installing TensorRT-LLM ]----------------"
pip3 install tensorrt_llm==0.8.0.dev2024012301 --extra-index-url https://pypi.nvidia.com