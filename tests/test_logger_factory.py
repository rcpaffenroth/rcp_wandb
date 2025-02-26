from rcpwandb.tracker_factory import TrackerFactory, TrackerFactoryFromConfig
import os
import pandas as pd

def test_tracker_factory_console():
    tracker = TrackerFactory(type='console', console_level='info')
    assert tracker is not None
    assert tracker._tracker_type == 'console'

def test_tracker_factory_wandb():
    os.environ['WANDB_MODE'] = 'disabled'
    tracker = TrackerFactory(type='wandb', console_level='info', 
                           project='test_project', entity='test_entity', name='test_name')
    assert tracker is not None
    assert tracker._tracker_type == 'wandb'

def test_tracker_factory_from_config_console():
    cfg = {'type': 'console', 'console_level': 'info'}
    tracker = TrackerFactoryFromConfig(cfg)
    assert tracker is not None
    assert tracker._tracker_type == 'console'

def test_tracker_factory_from_config_wandb():
    os.environ['WANDB_MODE'] = 'disabled'
    cfg = {'type': 'wandb', 'console_level': 'info', 
           'project': 'test_project', 'entity': 'test_entity', 'name': 'test_name'}
    tracker = TrackerFactoryFromConfig(cfg)
    assert tracker is not None
    assert tracker._tracker_type == 'wandb'

def test_tracker_factory_log_hyperparams():
    os.environ['WANDB_MODE'] = 'offline'
    tracker = TrackerFactory(type='wandb', 
                           console_level='info',
                           project='test')
    tracker.log_hyperparams({'a': 1, 'b': 2})
    tracker.finalize()

def test_tracker_factory_log_metric():
    os.environ['WANDB_MODE'] = 'offline'
    tracker = TrackerFactory(type='wandb', 
                           console_level='info',
                           project='test')
    tracker.log_metric('a', 1.0, 1)
    tracker.finalize()

def test_tracker_factory_log_metrics():
    os.environ['WANDB_MODE'] = 'offline'
    tracker = TrackerFactory(type='wandb', 
                           console_level='info',
                           project='test')
    tracker.log_metrics({'a': 1.0, 'b': 2.0}, 1)
    tracker.finalize()

def test_tracker_factory_log_dataframe():
    os.environ['WANDB_MODE'] = 'offline'
    tracker = TrackerFactory(type='wandb', 
                           console_level='info',
                           project='test')
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    tracker.log_dataframe('name', df)
    tracker.finalize()
