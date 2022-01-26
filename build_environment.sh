RED='\033[0;31m'

# make sure argument exist
if [ $# -eq 0 ]
  then
    echo -e "${RED}No arguments supplied, please run: source build_environment.sh YOUR_ENV_NAME  "
    exit -1
fi

# build environment & activate it
conda create --name $1 python==3.8.5
conda activate $1

# make sure conda env avtivated
if [ $CONDA_DEFAULT_ENV != $1 ]
  then
    echo -e "${RED}Your conda environment is not activated."
    exit -1
fi

# install packages
pip install -r requirements.txt

# Please modify the version that fits your GPU
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch