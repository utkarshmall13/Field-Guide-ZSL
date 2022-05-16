cd ..

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 0 --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 1 --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 2 --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 3 --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 4 --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 5 --num_shots $2 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 0 --fraction 0.8 --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 1 --fraction 0.8 --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 2 --fraction 0.8 --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 3 --fraction 0.8 --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 4 --fraction 0.8 --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 5 --fraction 0.8 --num_shots $2 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 0 --fraction 0.6  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 1 --fraction 0.6  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 2 --fraction 0.6  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 3 --fraction 0.6  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 4 --fraction 0.6  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 5 --fraction 0.6  --num_shots $2 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 0 --fraction 0.4  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 1 --fraction 0.4  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 2 --fraction 0.4  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 3 --fraction 0.4  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 4 --fraction 0.4  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 5 --fraction 0.4  --num_shots $2 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 0 --fraction 0.2  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 1 --fraction 0.2  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 2 --fraction 0.2  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 3 --fraction 0.2  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 4 --fraction 0.2  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 5 --fraction 0.2  --num_shots $2 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 0 --fraction 0.1  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 1 --fraction 0.1  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 2 --fraction 0.1  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 3 --fraction 0.1  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 4 --fraction 0.1  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 5 --fraction 0.1  --num_shots $2 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 0 --fraction 0.05  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 1 --fraction 0.05  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 2 --fraction 0.05  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 3 --fraction 0.05  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 4 --fraction 0.05  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 5 --fraction 0.05  --num_shots $2 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 0 --fraction 0.025  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 1 --fraction 0.025  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 2 --fraction 0.025  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 3 --fraction 0.025  --num_shots $2 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $1  --run 4 --fraction 0.025  --num_shots $2 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $1  --run 5 --fraction 0.025  --num_shots $2 &
wait

