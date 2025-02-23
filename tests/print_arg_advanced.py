import sys
from rcpwandb.logger_factory import LoggerFactory

logger = LoggerFactory(type='wandb', 
                       console_level='info', 
                       project='test')
logger.log_hyperparams({"sys.argv": sys.argv})

print(sys.argv)