import wandb
from typing import List, Tuple, Dict, Any
def baysean_search(search_name : str = "BayseanSearch", hyperparameters : Dict[str, Any] = None):
  sweep_config = {
    'method': 'bayes'
  }

  metric = {
    'name': 'EER',
    'goal': 'minimize'
  }

  sweep_config['metric'] = metric

  parameters_dict = {
    'p': {
      'distribution': 'uniform',
      'min': 0.1,
      'max': 1.0
    },
    'r': {
      'distribution': 'uniform',
      'min': 0.1,
      'max': 1.0
    },
    'alpha': {
      'distribution': 'uniform',
      'min': 0.9,
      'max': 1.2
    },
    'beta': {
      'distribution': 'uniform',
      'min': 1.0,
      'max': 4.0
    }
  }

  sweep_config['parameters'] = parameters_dict
  sweep_id = wandb.sweep(sweep_config, project=search_name)

  wandb.init()
  config = wandb.config

  print(config)


