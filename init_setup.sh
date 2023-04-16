conda create --prefix ./env python=3.7 -y
# source C:/Users/Asus/anaconda3/etc/profile.d/conda.sh # use your username instead of Asus
conda activate ./env
pip install -r requirements.txt
python setup.py install
conda env export > conda.yaml