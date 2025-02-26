import numpy as np
import pandas as pd
import plotly.express as px
import torch
from rcpwandb.tracker_factory import LoggerFactory

def demo():
    # Initialize the logger
    logger = LoggerFactory(type='wandb', console_level='info', project='test', name='demo')

    # Log hyperparameters
    hyperparams = {'learning_rate': 0.001, 'batch_size': 32}
    logger.log_hyperparams(hyperparams)

    # Log metrics
    for i in range(10):
        metrics = {'accuracy': np.cos(0.9*i), 'loss': 0.1*i}
        logger.log_metrics(metrics, step=i)
        logger.log_metric('single metric', np.sin(0.9*i), step=i)    

    # Log an image
    data = np.random.rand(100, 100)
    logger.log_image('random_image', data)

    # Log a plot
    fig = px.scatter(x=np.random.rand(100), y=np.random.rand(100))
    logger.log_plot('scatter_plot', fig)

    # Log a 3d plot!
    df = px.data.iris()
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')
    logger.log_plot('iris', fig)

    # Create a sample dataframe with an image
    data = np.random.rand(100, 100)
    bw_image = logger.make_image(data)
    data = np.random.rand(100, 100, 3)
    color_image = logger.make_image(data)

    data = {
        'id': [1, 2],
        'bw_image': [bw_image, bw_image],
        'color_image': [color_image, color_image]
    }
    df = pd.DataFrame(data)
    logger.log_dataframe('sample_table', df)

    # Save a model
    model = torch.nn.Linear(10, 1)
    model_path = 'dummy_model.pt'
    torch.save(model.state_dict(), model_path)
    logger.save_model(model_path, 'dummy_model')

    # Finalize the logger
    logger.finalize()

if __name__ == "__main__":
    demo()