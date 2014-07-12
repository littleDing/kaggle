tag=$1
nohup python solutions.py $tag 2>stderr.$tag >stdout.$tag &
