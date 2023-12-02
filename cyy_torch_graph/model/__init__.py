import copy

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       ModelType)
from cyy_torch_toolbox.model import globel_model_factory
from cyy_torch_toolbox.model.repositary import get_model_info


def get_model(
    name: str, dataset_collection: DatasetCollection, model_kwargs: dict
) -> dict:
    model_constructors = get_model_info().get(DatasetType.Graph, {})
    model_constructor_info = model_constructors.get(name.lower(), {})
    if not model_constructor_info:
        raise NotImplementedError(
            f"unsupported model {name}, supported models are "
            + str(model_constructors.keys())
        )

    final_model_kwargs: dict = {}
    match dataset_collection.dataset_type:
        case DatasetType.Graph:
            if "num_features" not in model_kwargs:
                final_model_kwargs[
                    "num_features"
                ] = dataset_collection.get_original_dataset(
                    phase=MachineLearningPhase.Training
                ).num_features

    final_model_kwargs |= model_kwargs
    model_type = ModelType.Classification
    if model_type in (ModelType.Classification, ModelType.Detection):
        if "num_classes" not in final_model_kwargs:
            final_model_kwargs["num_classes"] = dataset_collection.label_number  # E:
            get_logger().debug("detect %s classes", final_model_kwargs["num_classes"])
        else:
            assert (
                final_model_kwargs["num_classes"] == dataset_collection.label_number
            )  # E:
    if model_type == ModelType.Detection:
        final_model_kwargs["num_classes"] += 1
    final_model_kwargs["num_labels"] = final_model_kwargs["num_classes"]
    # use_checkpointing = model_kwargs.pop("use_checkpointing", False)
    while True:
        try:
            model = model_constructor_info["constructor"](**final_model_kwargs)
            get_logger().debug(
                "use model arguments %s for model %s",
                final_model_kwargs,
                model_constructor_info["name"],
            )
            res = {"model": model}
            return res
        except TypeError as e:
            retry = False
            for k in copy.copy(final_model_kwargs):
                if k in str(e):
                    get_logger().debug("%s so remove %s", e, k)
                    final_model_kwargs.pop(k)
                    retry = True
                    break
            # if not retry:
            #     if "pretrained" in str(e) and not model_kwargs["pretrained"]:
            #         model_kwargs.pop("pretrained")
            #         retry = True
            if not retry:
                raise e


globel_model_factory.register(DatasetType.Graph, get_model)
