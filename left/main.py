#! /usr/bin/env python3
import pickle
import logging
import sys
import torch
from pathlib import Path
import copy
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import jax

from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.algorithms import run_algorithm
from circuits_benchmark.commands.train import train
from circuits_benchmark.commands.evaluation import evaluation

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
logging.basicConfig(level=logging.ERROR)


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup_ddp():
    dist.destroy_process_group()


def task(args):
  if args.command == "run":
    run_algorithm.run(args)
  elif args.command == "train":
    train.run(args)
  elif args.command == "eval":
    evaluation.run(args)


if __name__ == "__main__":
  parser = build_main_parser()

  args, _ = parser.parse_known_args(sys.argv[1:])

  # get the admissible cases
  admissible = pickle.load(open("admissible_tasks.pkl", "rb"))

  # rank, world_size = setup_ddp()

  for case in list(admissible.values())[::-1]:
    for seed in range(0, 10):
      args_cs = copy.copy(args)
      # args_cs.device = f"cuda:{rank}"
      args_cs.seed = seed
      args_cs.indices= case.get_name()
      args_cs.output_dir = str(Path(args_cs.output_dir) / f"c{case.get_name()}-s{seed}")
      task(args_cs)

  cleanup_ddp()