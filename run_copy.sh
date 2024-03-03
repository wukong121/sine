python main.py --gpu 0 --dataset ml1m --category_num 2 --topic_num 10 --experiment 1 --epoch 30 --beta 0.1 
python main.py --gpu 0 --dataset ml1m --category_num 2 --topic_num 10 --experiment 0 --epoch 30 

python main.py --gpu 0 --dataset book --category_num 4 --hidden_size 128 --model_type DNN --experiment -1 --epoch 15
python main.py --gpu 1 --dataset book --category_num 4 --hidden_size 128 --model_type GRU4REC --experiment -1 --epoch 15
python main.py --gpu 2 --dataset book --category_num 4 --hidden_size 128 --model_type MIND --experiment -1 --epoch 15

python main.py --gpu 2 --dataset book --category_num 4 --hidden_size 128 --topic_num 500 --model_type SINE --experiment 0 --temperature 0.4 --epoch 15
python main.py --gpu 0 --dataset book --category_num 4 --hidden_size 128 --topic_num 500 --model_type SINE --experiment 0 --temperature 0.3 --epoch 15

python main.py --gpu 0 --dataset ml1m --category_num 2 --hidden_size 128 --model_type DNN --experiment -1 --epoch 20 
python main.py --gpu 1 --dataset ml1m --category_num 2 --hidden_size 128 --model_type GRU4REC --experiment -1 --epoch 20 
python main.py --gpu 2 --dataset ml1m --category_num 2 --hidden_size 128 --model_type MIND --experiment -1 --epoch 20 

python test.py --gpu 0 --dataset ml1m --category_num 2 --topic_num 10 --experiment 1 --epoch 30 --beta 0.1 