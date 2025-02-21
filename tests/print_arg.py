import sys
import wandb

wandb.init()
wandb.log({"sys.argv": sys.argv})

print(sys.argv)