cd ../

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+0 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+0 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+0 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+0 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+0 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+0 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+4 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+4 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+4 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+4 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+4 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+4 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+8 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+8 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+8 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+8 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+8 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+8 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+12 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+12 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+12 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+12 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+12 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+12 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+16 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+16 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+16 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+16 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+16 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+16 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+20 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+20 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+20 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+20 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+20 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+20 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+24 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+24 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+24 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+24 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+24 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+24 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+32 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+32 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+32 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+32 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+32 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+32 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+40 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+40 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+40 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+40 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+40 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+40 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+56 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+56 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+56 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+56 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+56 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+56 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+72 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+72 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+72 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+72 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+72 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+72 --train_mode $1 --num_shots $4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 0 --test_mode $2+85 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 1 --test_mode $2+85 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 2 --test_mode $2+85 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 3 --test_mode $2+85 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=0 python3 inference.py --dataset $3  --run 4 --test_mode $2+85 --train_mode $1 --num_shots $4 &
CUDA_VISIBLE_DEVICES=1 python3 inference.py --dataset $3  --run 5 --test_mode $2+85 --train_mode $1 --num_shots $4 &
wait
