from loguru import logger as logging
from icecream import ic
import sys

# This should be the only place in the code where this import happen
import wandb

# We need to be careful about imports and this is a lib
# that helps with that.
import importlib

# This is a little trick to make ic work with logging.  This is useful
# since ic has nice formatting.
def info(s):
    """Debug function that works with ic

    Args:
        s (str): The thing to log
    """
    logging.info(s)
ic.configureOutput(outputFunction=info)

# The global logger object
logger = None

class LoggerFacade:
    def __init__(self, logger, logger_type, console_level='info'):
        self._logger = logger
        self._logger_type = logger_type

        self.console_logger = logging
        # This bears some explanation.  We want to be able to control the
        # the logging level even when the logger is running in a separate
        # process.  For example, this happens when the logger is run
        # through Dask.
        if console_level == 'debug':
            logging.add(sink=sys.stdout, level='DEBUG')
        elif console_level == 'info':
            logging.add(sink=sys.stdout, level='INFO')
        elif console_level == 'error':
            logging.add(sink=sys.stdout, level='ERROR')
        else:
            assert False, f"Unknown console level {console_level}"    
        self.console_logger.info(f"logger_type {self._logger_type}")

    def make_image(self, image):
        # This function is used to convert an image into a format 
        # that can be displayed in wandb.
        return wandb.Image(image)

    def log_image(self, name, image, step=None):
        if self._logger_type == 'wandb':
            self._logger.log({name: wandb.Image(image)}, step=step)
        self.console_logger.debug(f"name {name} step {step}")

    # This is intended to be used for putting a plotly figure into
    # a pandas dataframe.  This is not currently used, so I leave it here
    # for future reference.
    # def make_plot(self, name, fig):
    #     # This function is used to convert a figure into a format
    #     # that can be displayed in wandb.
    #     # Create path for Plotly figure
    #     path = pathlib.Path(wandb.run.dir) / f"{name}.html"
    #     # Write Plotly figure to HTML
    #     fig.write_html(path, auto_play = False) 
    #     return wandb.HTML(fig)

    def log_plot(self, name, fig, step=None):
        if self._logger_type == 'wandb':
            self._logger.log({name: wandb.Plotly(fig)}, step=step)
        self.console_logger.debug(f"name {name} step {step}")

    def log_video(self, name, path_to_video, step=None):
        if self._logger_type == 'wandb':
            self._logger.log({name: wandb.Video(path_to_video)}, step=step)
        self.console_logger.debug(f"name {name} step {step}")

    def log_hyperparams(self, params):
        if self._logger_type == 'wandb':
            self._logger.config.update(params)
        self.console_logger.debug(f"params {params}")

    def log_metric(self, name, value, step=None):
        if self._logger_type == 'wandb':
            self._logger.log({name: value}, step=step)
        self.console_logger.debug(f"name {name} value {value} step {step}")

    def log_metrics(self, metrics, step=None):
        # metrics is a dictionary of metric names and values
        if self._logger_type == 'wandb':
            self._logger.log(metrics, step=step)
        self.console_logger.debug(f"metrics {metrics} step {step}")

    def log_dataframe(self, name, df, step=None):
        if self._logger_type == 'wandb':
            self._logger.log({name: df}, step=step)
        else:
            self.console_logger.warning(f"log_dataframe not implemented for logger type {self._logger_type}")  

    def finalize(self):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        # This doesn't seem to break anything, though it does lead to error messages.
        # Also, it doesn't seen to be necessary.
        if self._logger_type == 'wandb':
            wandb.finish()

    def save_model(self, model_path, model_name):
        if self._logger_type == 'wandb':
            # path = pathlib.Path(wandb.run.dir) / save_model_name
            # Save the model to wandb
            # torch.save(model.state_dict(), path)
            # Save as artifact for version control.
            artifact = wandb.Artifact(model_name, type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
        else:
            self.console_logger.warning(f"save_model not implemented for logger type {self._logger_type}") 

    def get_sb3_wandb_callback(self):
        """Get the Stable Baselines3 WandbCallback
            This bears some explanation.  The WandbCallback is a callback
            that can be used with the Stable Baselines3 library.  It logs
            the metrics from the training to wandb.  This is useful for
            tracking the progress of the training.  However, to include it 
            requires many heavy dependencies.  This function is a way to
            include the WandbCallback without including the dependencies here.
        """
        try:
            sb3_import = importlib.import_module('wandb.integration.sb3')
        except ImportError:
            self.console_logger.warning("wandb.integration.sb3 not available")
            return None
        return sb3_import.WandbCallback

def LoggerFactory(type='console', console_level='info',
                  project=None, entity=None, name=None,
                  model=None, default_root_dir=None):
    cfg = {'type': type, 'console_level': console_level, 'name': name, 'project': project, 'entity': entity}
    return LoggerFactoryFromConfig(cfg, model, default_root_dir)

def LoggerFactoryFromConfig(cfg=None, model=None, default_root_dir=None):
    global logger
    if logger is not None:
        logger.console_logger.warning("Logger already created")
        return logger

    if cfg is None:
        cfg = {}
        cfg['type'] = 'console'
        cfg['console_level'] = 'info'

    if cfg['type'] == 'wandb':
        # RCP: Note the following two lines appear in
        # https://docs.wandb.ai/guides/track/log/distributed-training
        # #hanging-at-the-beginning-of-training
        # But they don't seem to be necessary for the current version of wandb.
        # I include them anyway just in case.
        # os.environ["WANDB_START_METHOD"] = "thread"
        wandb.setup()
        wandb.init(project=cfg['project'],
                   entity=cfg['entity'],
                   name=cfg['name'],
                   dir=default_root_dir,
                   reinit=True)
        if model:
            wandb.watch(model, log_graph=False, log_freq=100)
        logger = LoggerFacade(wandb, 'wandb', cfg['console_level'])
    elif cfg['type'] == 'wandb_gym':
        # RCP: Note the following two lines appear in
        # https://docs.wandb.ai/guides/track/log/distributed-training
        # #hanging-at-the-beginning-of-training
        # But they don't seem to be necessary for the current version of wandb.
        # I include them anyway just in case.
        # os.environ["WANDB_START_METHOD"] = "thread"
        wandb.setup()
        wandb.init(project=cfg['project'],
                   entity=cfg['entity'],
                   name=cfg['name'],
                   dir=default_root_dir,
                   reinit=True,
                   sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                   # FIXME: video uploading is currently broken 2/2/2025
                   # https://github.com/DLR-RM/stable-baselines3/issues/2055
                   monitor_gym=True,  # auto-upload the videos of agents playing the game
                   save_code=True)  # optional
        if model:
            wandb.watch(model, log_graph=False, log_freq=100)
        logger = LoggerFacade(wandb, 'wandb', cfg['console_level'])
    elif cfg['type'] == 'console':
        logger = LoggerFacade(None, 'console', cfg['console_level'])
    else:
        raise ValueError(f"Unknown logger type: {cfg['type']}")
    
    logger.console_logger.info(f"All logging it going to {default_root_dir}")
    return logger

