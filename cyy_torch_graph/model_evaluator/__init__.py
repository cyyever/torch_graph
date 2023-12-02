from cyy_torch_toolbox.ml_type import DatasetType
from cyy_torch_toolbox.model import global_model_evaluator_factory

from .graph import GraphModelEvaluator

global_model_evaluator_factory.register(DatasetType.Graph, GraphModelEvaluator)
