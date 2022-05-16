# training a single model
python3 train_vae.py --dataset CUB  --run 0

# inferring using sibling variance with 10 question (attributes)
python3 inference.py --dataset CUB  --run 0 --test_mode interactive_siblings+10 --train_mode standard --num_shots 0

# inferring using sibling variance with 20 question (attributes)
python3 inference.py --dataset CUB  --run 0 --test_mode interactive_siblings+20 --train_mode standard --num_shots 0

# inferring using representation change with 6 question (attributes)
python3 inference.py --dataset CUB  --run 0 --test_mode interalllatent+6 --train_mode standard --num_shots 0


# inferring 1-shot learning using image based learning.
python3 inference.py --dataset CUB  --run 0 --test_mode oneshotactive+6 --train_mode standard --num_shots 1

# inferring 1-shot learning using sibling varians.
python3 inference.py --dataset CUB  --run 0 --test_mode interactive_siblings+6 --train_mode standard --num_shots 1
