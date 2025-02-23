#!/usr/bin/env python

import click
from click.testing import CliRunner
from icecream import ic
import sys
from pathlib import Path
import subprocess
import tempfile
import yaml
import wandb
import pandas as pd
import uuid

def sanity_checks():
    """
    Perform sanity checks to ensure the script is running in a virtual environment.

    This function checks the current Python interpreter path using `sys.executable` and `pathlib`.
    If the interpreter is found to be in one of the common system directories
    (`/usr/bin`, `/usr/local/bin`, or `~/minimama/bin`), it raises a `ValueError` indicating
    that the script is expected to run in a virtual environment. Otherwise, it logs the
    interpreter path using the `ic` function.

    Raises:
        ValueError: If the Python interpreter is found in one of the common system directories.
    """

    # get the current Python interpreter path using pathlib from sys.executable
    python_interpreter = Path(sys.executable)
    if python_interpreter.parent in [
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path.home() / "minimama/bin",
    ]:
        raise ValueError(
            f"Running with {python_interpreter}, expected a virtual environment."
        )


def run(command, capture_output=True):
    """
    Run a shell command using `subprocess.run`.

    This function takes a shell command as input and runs it using `subprocess.run`.
    It uses the `shell=True` argument to run the command in a subshell.
    The function captures the output of the command and returns it as a string.

    Args:
        command (str): The shell command to run.

    Returns:
        str: The output of the shell command.
    """
    result = subprocess.run(
        command, shell=True, capture_output=capture_output, text=True
    )
    return result.stdout


@click.group()
def cli():
    """The main entry point for RCP's modifciation of the wandb commands."""
    pass


@cli.command()
@click.argument("sweep_file", default=sys.stdin)
def sweep(sweep_file):
    """Run a sweep using the specified sweep file.
    This does the same thing at the `wandb sweep` command, but with some additional checks, and
    with just the SWEEP_ID as the output.
    """
    # Perform sanity checks to ensure everything is ok
    sanity_checks()

    # check that the sweep file exists
    if not Path(sweep_file).exists():
        raise FileNotFoundError(f"File {sweep_file} not found.")

    with tempfile.NamedTemporaryFile() as tmp_file:
        run(f"wandb sweep {sweep_file} > {tmp_file.name} 2>&1")
        tmp_file.seek(0)
        # Get the sweep_id from the last entry of the line of the output
        # NOTE: the outout of readlines is a byte string, so we need to decode it
        sweep_id = tmp_file.readlines()[-1].decode().strip().split()[-1]
        print(sweep_id)


@cli.command()
@click.argument("sweep_id", default="")
def agent(sweep_id):
    """Run a sweep locally using the specified sweep file.
    This does the same thing at the `wandb agent` command, but is chainable using stdout.
    Also, by default, it runs in the background and silently logs to a file.
    """
    # Perform sanity checks to ensure everything is ok
    sanity_checks()
    if sweep_id == "":
        sweep_id = sys.stdin.read().strip()
    _agent(sweep_id)

def _agent(sweep_id):
    log = f"log_{uuid.uuid4()}.txt"
    ic(sweep_id, log)
    run(f"wandb agent {sweep_id} > {log} 2>&1 &", capture_output=False)

@cli.command()
@click.argument("path")
@click.option(
    "-y", "--yes", default=False, is_flag=True, help="Proceed without verification"
)
def deleteruns(path, yes):
    """
    Delete runs from a project.

    This function performs sanity checks and then deletes runs associated with a given project path.
    Related wandb documentation https://docs.wandb.ai/ref/python/public-api/api/

    Args:
        path (str): The path to the project from which runs are to be deleted.

    Returns:
        None
    """
    if not yes:
        click.confirm(
            f"Are you sure you want to delete all runs in path {path}?", abort=True
        )
    # perhaps this is paranoid, but I want to only include wandb if I need it
    import wandb

    # Perform sanity checks to ensure everything is ok
    sanity_checks()
    for wandb_run in wandb.Api().runs(path=path):
        ic("deleting run", wandb_run)
        wandb_run.delete()


@cli.command()
@click.argument("sweep_id", default="")
@click.option(
    "-w",
    "--worker_file",
    type=click.Path(),
    default="run_sweep.yml",
    help="The yaml file with the worker configuration.")
def multiagent(sweep_id, worker_file):
    """Run a sweep locally using the specified sweep file.
    This does the same thing at the `wandb agent` command, but is chainable using stdout.
    Also, by default, it runs in the background and silently logs to a file.
    """
    # Perform sanity checks to ensure everything is ok
    sanity_checks()

    # if sweep_id == "":
    #     sweep_id = sys.stdin.read().strip()

    # Look for a yaml file with the worker configuration such as
    # workers:
    #   - host: beelink1
    #     directory: /home/rcpaffenroth/projects/2_research/inn_survey
    #     branch: main
    #   - host: beelink2
    #     directory: /home/rcpaffenroth/projects/2_research/inn_survey
    #     branch: main

    work_dict = yaml.safe_load(open(worker_file, "r"))
    if sweep_id == "":
        sweep_id = sys.stdin.read().strip()

    ic(work_dict)
    for worker in work_dict["workers"]:
        ic(worker)
        if worker['worker'] == "local":
            for i in range(worker['number']):
                _agent(sweep_id)

@cli.command()
@click.option("--entity", default="rcpaffenroth-wpi", help="The entity to use.")
@click.option("--project", default="inn-survey-test", help="The project to use.")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="project.csv",
    help="The output file to save the data to.",
)
def export(entity, project, output):
    """
    Export the summary, configuration, and name of all runs in a given Weights & Biases project to a CSV file.

    Args:
        entity (str): The entity (user or team) under which the project is hosted.
        project (str): The name of the project from which to export runs.

    Returns:
        None: The function saves the exported data to a CSV file named 'project.csv'.
    """

    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    run_list = []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        run_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        run_list[-1].update({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        run_list[-1]['name'] = run.name

    runs_df = pd.DataFrame(run_list)
    runs_df.to_csv(output)
    return runs_df


def test_main():
    runner = CliRunner()
    result = runner.invoke(sweep)
    assert result.exit_code == 0


if __name__ == "__main__":
    cli()
