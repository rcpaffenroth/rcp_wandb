from rcp_wandb.logger_factory import LoggerFactory, LoggerFactoryFromConfig
import os
import pandas as pd

def test_logger_factory_console():
    logger = LoggerFactory(type='console', console_level='info')
    assert logger is not None
    assert logger._logger_type == 'console'

def test_logger_factory_wandb():
    os.environ['WANDB_MODE'] = 'disabled'
    logger = LoggerFactory(type='wandb', console_level='info', 
                           project='test_project', entity='test_entity', name='test_name')
    assert logger is not None
    assert logger._logger_type == 'wandb'

def test_logger_factory_from_config_console():
    cfg = {'type': 'console', 'console_level': 'info'}
    logger = LoggerFactoryFromConfig(cfg)
    assert logger is not None
    assert logger._logger_type == 'console'

def test_logger_factory_from_config_wandb():
    os.environ['WANDB_MODE'] = 'disabled'
    cfg = {'type': 'wandb', 'console_level': 'info', 
           'project': 'test_project', 'entity': 'test_entity', 'name': 'test_name'}
    logger = LoggerFactoryFromConfig(cfg)
    assert logger is not None
    assert logger._logger_type == 'wandb'

def test_logger_factory_log_hyperparams():
    os.environ['WANDB_MODE'] = 'offline'
    logger = LoggerFactory(type='wandb', 
                           console_level='info',
                           project='test')
    logger.log_hyperparams({'a': 1, 'b': 2})
    logger.finalize()

def test_logger_factory_log_metric():
    os.environ['WANDB_MODE'] = 'offline'
    logger = LoggerFactory(type='wandb', 
                           console_level='info',
                           project='test')
    logger.log_metric('a', 1.0, 1)
    logger.finalize()

def test_logger_factory_log_metrics():
    os.environ['WANDB_MODE'] = 'offline'
    logger = LoggerFactory(type='wandb', 
                           console_level='info',
                           project='test')
    logger.log_metrics({'a': 1.0, 'b': 2.0}, 1)
    logger.finalize()

def test_logger_factory_log_dataframe():
    os.environ['WANDB_MODE'] = 'offline'
    logger = LoggerFactory(type='wandb', 
                           console_level='info',
                           project='test')
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    logger.log_dataframe('name', df)
    logger.finalize()
