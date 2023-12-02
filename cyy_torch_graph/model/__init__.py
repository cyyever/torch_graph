import functools

from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.factory import Factory
from cyy_torch_toolbox.ml_type import DatasetType, MachineLearningPhase
from cyy_torch_toolbox.model import global_model_factory
from cyy_torch_toolbox.model.repositary import get_model_info


def get_model(
    model_constructor_info: dict, dataset_collection: DatasetCollection, **kwargs
) -> dict:
    final_model_kwargs: dict = {}
    match dataset_collection.dataset_type:
        case DatasetType.Graph:
            if "num_features" not in kwargs:
                final_model_kwargs[
                    "num_features"
                ] = dataset_collection.get_original_dataset(
                    phase=MachineLearningPhase.Training
                ).num_features

    final_model_kwargs |= kwargs
    return model_constructor_info["constructor"](**final_model_kwargs)


model_constructors = get_model_info().get(DatasetType.Graph, {})
for name, model_constructor_info in model_constructors.items():
    if DatasetType.Graph not in global_model_factory:
        global_model_factory[DatasetType.Graph] = Factory()
    global_model_factory[DatasetType.Graph].register(
        name, functools.partial(get_model, model_constructor_info)
    )
