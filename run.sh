for seed in 0 1 2 3 4; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python train.py --env=LunarLander-v3 --seed=$seed > logs_seed_${seed}.log 2>&1 &

  while [ "$(jobs -r | wc -l)" -ge 2 ]; do
    sleep 5
  done
done
wait