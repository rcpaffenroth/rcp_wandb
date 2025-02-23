# rcp_wandb
RCPs extensions to Weights and Biases

## Usage examples

For example, to do a local sweep and then clean up the runs
```bash
wandb sweep run_sweep.yml 
wandb agent COPIED_FROM_ABOVE
./rcp_wandb sweep run_sweep.yml | ./rcp_wandb agent
./rcp_wandb deleteruns rcpaffenroth-wpi/inn-survey-test
./rcp_wandb multiagent -w run_sweep.yml foo/foo/foo
./rcp_wandb sweep run_sweep.yml | ./rcp_wandb multiagent -w run_sweep.yml
```

## Tests

```bash
wandb sweep run_sweep_basic.yml 
wandb agent COPIED_FROM_ABOVE
rcp_wandb deleteruns rcpaffenroth-wpi/test
```

```bash
rcp_wandb sweep run_sweep_basic.yml | rcp_wandb agent && rcp_wandb deleteruns -y rcpaffenroth-wpi/test
```

```bash
rcp_wandb sweep run_sweep_advanced.yml | rcp_wandb agent && rcp_wandb deleteruns -y rcpaffenroth-wpi/test
```

```bash
rcp_wandb sweep run_sweep_advanced.yml | rcp_wandb multiagent -w run_sweep_advanced.yml && rcp_wandb deleteruns -y rcpaffenroth-wpi/test
```