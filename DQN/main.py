# environment
import gymnasium

# jax imports
import numpy as np
import jax 
from jax import numpy as jnp

# basic imports
import argparse
import types

def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog = "DQN implementation")
    parser.add_argument('--test_parser', type=bool, default=True)
    return parser

def main() -> None:
    args = args_parser().parse_args()
    print(args.test_parser)
    
    

if __name__ == "__main__":
    main()