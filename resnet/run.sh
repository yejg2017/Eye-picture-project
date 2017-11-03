workdir="/home/ye/user/yejg/project/ResNet"
exe="train_and_val.py"
result="result/"
data="/home/ye/user/yejg/database/eye_jpg/train/"
model_save="model_path/"

python $workdir/$exe -d $data  -o $workdir/$result -s $workdir/$model_save --gw 224 --gh 224 -w 0.0005 --lr 0.0001 -b 64 -M 50000 
