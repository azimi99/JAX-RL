parallel -j 2 \
'XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py --env={1} --seed={2} > logs_{1}_seed_{2}.log 2>&1' \
::: CartPole-v1 LunarLander-v3 \
::: 0 1 2 3 4