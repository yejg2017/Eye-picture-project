workdir="/home/ye/user/yejg/project/Eye_densenet_1/"
code_exe="/home/ye/user/yejg/project/Eye_densenet_1/train.py"
model_path="/home/ye/user/yejg/project/Eye_densenet_1/model_path/"
data_path="/home/ye/user/yejg/database/eye_jpg/train/"

python $code_exe -D $data_path  -b 64 -d 16 -M 50000 -H 224 -W 224 -r 0.25 -f $model_path  
