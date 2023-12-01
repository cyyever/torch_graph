import cyy_torch_graph  # noqa: F401
from cyy_torch_toolbox.default_config import Config
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def test_graph_training() -> None:
    config = Config(dataset_name="Yelp", model_name="OneGCN")
    config.trainer_config.hook_config.debug = True
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    # trainer.model_with_loss.compile_model()
    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
    )
    trainer.train()
