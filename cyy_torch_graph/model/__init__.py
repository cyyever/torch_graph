import functools

from cyy_torch_toolbox import (DatasetCollection, DatasetType,
                               MachineLearningPhase)
from cyy_torch_toolbox.factory import Factory
from cyy_torch_toolbox.model import (create_model,
                                     global_model_evaluator_factory,
                                     global_model_factory)
from cyy_torch_toolbox.model.repositary import get_model_info

from .evaluator import GraphModelEvaluator

__all__ = ["GraphModelEvaluator"]
global_model_evaluator_factory.register(DatasetType.Graph, GraphModelEvaluator)


def get_model(
    model_constructor_info: dict, dataset_collection: DatasetCollection, **kwargs
) -> dict:
    final_model_kwargs: dict = kwargs
    if "num_features" not in kwargs:
        final_model_kwargs["num_features"] = dataset_collection.get_original_dataset(
            phase=MachineLearningPhase.Training
        ).num_features
    return {
        "model": create_model(
            model_constructor_info["constructor"], **final_model_kwargs
        )
    }


model_constructors = get_model_info().get(DatasetType.Graph, {})
for name, constructor_info in model_constructors.items():
    if DatasetType.Graph not in global_model_factory:
        global_model_factory[DatasetType.Graph] = Factory()
    global_model_factory[DatasetType.Graph].register(
        name, functools.partial(get_model, constructor_info)
    )
