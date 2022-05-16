cd ..

CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 0 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 1 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 2 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 3 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 4 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 5 &
wait

CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 0 --fraction 0.8 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 1 --fraction 0.8 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 2 --fraction 0.8 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 3 --fraction 0.8 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 4 --fraction 0.8 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 5 --fraction 0.8 &
wait

CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 0 --fraction 0.6 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 1 --fraction 0.6 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 2 --fraction 0.6 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 3 --fraction 0.6 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 4 --fraction 0.6 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 5 --fraction 0.6 &
wait

CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 0 --fraction 0.4 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 1 --fraction 0.4 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 2 --fraction 0.4 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 3 --fraction 0.4 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 4 --fraction 0.4 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 5 --fraction 0.4 &
wait

CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 0 --fraction 0.2 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 1 --fraction 0.2 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 2 --fraction 0.2 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 3 --fraction 0.2 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 4 --fraction 0.2 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 5 --fraction 0.2 &
wait

CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 0 --fraction 0.1 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 1 --fraction 0.1 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 2 --fraction 0.1 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 3 --fraction 0.1 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 4 --fraction 0.1 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 5 --fraction 0.1 &
wait

CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 0 --fraction 0.05 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 1 --fraction 0.05 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 2 --fraction 0.05 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 3 --fraction 0.05 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 4 --fraction 0.05 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 5 --fraction 0.05 &
wait

CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 0 --fraction 0.025 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 1 --fraction 0.025 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 2 --fraction 0.025 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 3 --fraction 0.025 &
CUDA_VISIBLE_DEVICES=0 python3 train_vae.py --dataset $1  --run 4 --fraction 0.025 &
CUDA_VISIBLE_DEVICES=1 python3 train_vae.py --dataset $1  --run 5 --fraction 0.025 &
wait

