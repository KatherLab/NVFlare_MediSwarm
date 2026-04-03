# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from nvflare.app_common.abstract.fl_model import FLModel, MetaKey
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.client.api import clear, get_config, init, is_evaluate, is_submit_model, is_train, receive, send
from nvflare.client.config import ConfigKey
from nvflare.fuel.utils import fobs

from .callbacks import RestoreState

FL_META_KEY = "__fl_meta__"


def patch(
    trainer: pl.Trainer, restore_state: bool = True, load_state_dict_strict: bool = True, update_fit_loop: bool = True
):
    """Patches the PyTorch Lightning Trainer for usage with NVFlare.

    Args:
        trainer: the PyTorch Lightning trainer.
        restore_state: whether to restore optimizer and learning rate scheduler states.
            Defaults to `True`.
        load_state_dict_strict: exposes `strict` argument of `torch.nn.Module.load_state_dict()`
            used to load the received model. Defaults to `True`.
            See https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict for details.
        update_fit_loop: whether to increase `trainer.fit_loop.max_epochs` and `trainer.fit_loop.epoch_loop.max_steps` each FL round.
            Defaults to `True` which is suitable for most PyTorch Lightning applications.

    Example:

        Normal usage:

        .. code-block:: python

            trainer = Trainer(max_epochs=1)
            flare.patch(trainer)


        Advanced usage:

        If users want to pass additional information to FLARE server side via the lightning API,
        they will need to set the information inside the attributes called ``__fl_meta__`` in their LightningModule.

        .. code-block:: python

            class LitNet(LightningModule):
                def __init__(self):
                    super().__init__()
                    self.save_hyperparameters()
                    self.model = Net()
                    self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
                    self.valid_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
                    self.__fl_meta__ = {"CUSTOM_VAR": "VALUE_OF_THE_VAR"}

    """
    fobs.register(TensorDecomposer)
    callbacks = trainer.callbacks
    if isinstance(callbacks, Callback):
        callbacks = [callbacks]
    elif not isinstance(callbacks, list):
        callbacks = []

    if not any(isinstance(cb, FLCallback) for cb in callbacks):
        fl_callback = FLCallback(
            rank=trainer.global_rank, load_state_dict_strict=load_state_dict_strict, update_fit_loop=update_fit_loop
        )
        callbacks.append(fl_callback)

    if restore_state and not any(isinstance(cb, RestoreState) for cb in callbacks):
        callbacks.append(RestoreState())

    trainer.callbacks = callbacks


class FLCallback(Callback):
    def __init__(self, rank: int = 0, load_state_dict_strict: bool = True, update_fit_loop: bool = True):
        """FL callback for lightning API.

        Args:
            rank: global rank of the PyTorch Lightning trainer.
            load_state_dict_strict: exposes `strict` argument of `torch.nn.Module.load_state_dict()`
                used to load the received model. Defaults to `True`.
                See https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict for details.
            update_fit_loop: whether to increase `trainer.fit_loop.max_epochs` and `trainer.fit_loop.epoch_loop.max_steps` each FL round.
                Defaults to `True` which is suitable for most PyTorch Lightning applications.
        """
        super(FLCallback, self).__init__()
        init(rank=str(rank))
        self.train_with_evaluation = get_config().get(ConfigKey.TASK_EXCHANGE, {}).get(ConfigKey.TRAIN_WITH_EVAL, False)
        self.current_round = None
        self.metrics = None
        self.total_local_epochs = 0
        self.total_local_steps = 0
        self.max_epochs_per_round = None
        self.max_steps_per_round = None
        self.rank = rank
        self._is_training = False
        self._is_evaluation = False
        self._is_submit_model = False
        self._load_state_dict_strict = load_state_dict_strict
        self._update_fit_loop = update_fit_loop

        self.logger = logging.getLogger(self.__class__.__name__)

    def reset_state(self, trainer):
        """Resets the state.

        If the next round of federated training needs to reuse the same callback
        instance, the reset_state() needs to be called first
        Not only resets the states, also sets states for next round
        """
        # set states for next round
        if self.current_round is not None:
            if self.max_epochs_per_round is None:
                if trainer.max_epochs and trainer.max_epochs > 0:
                    self.max_epochs_per_round = trainer.max_epochs
                if trainer.max_steps and trainer.max_steps > 0:
                    self.max_steps_per_round = trainer.max_steps

            # record total local epochs/steps
            self.total_local_epochs = trainer.current_epoch
            self.total_local_steps = trainer.estimated_stepping_batches

            # for next round
            trainer.num_sanity_val_steps = 0  # Turn off sanity validation steps in following rounds of FL

            if self._update_fit_loop:
                if self.total_local_epochs and self.max_epochs_per_round is not None:
                    trainer.fit_loop.max_epochs = self.max_epochs_per_round + self.total_local_epochs
                if self.total_local_steps and self.max_steps_per_round is not None:
                    trainer.fit_loop.epoch_loop.max_steps = self.max_steps_per_round + self.total_local_steps

        # resets attributes
        self.metrics = None
        clear()

    def on_train_start(self, trainer, pl_module):
        # receive the global model and update the local model with global model
        self._receive_and_update_model(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, FL_META_KEY):
            fl_meta = getattr(pl_module, FL_META_KEY)
            if not isinstance(fl_meta, dict):
                raise RuntimeError(f"The {FL_META_KEY} needs to be a dictionary")
        else:
            fl_meta = {}
        if MetaKey.NUM_STEPS_CURRENT_ROUND not in fl_meta:
            fl_meta[MetaKey.NUM_STEPS_CURRENT_ROUND] = trainer.estimated_stepping_batches
        if self._is_training:
            model = FLModel(params=pl_module.cpu().state_dict(), meta=fl_meta)
            if self.train_with_evaluation:
                if self.metrics is None:
                    raise RuntimeError(
                        "train with evaluation missing training metrics, please remember to call validate."
                    )
                model.metrics = self.metrics
            self._send_model(model)
            self.reset_state(trainer)

    def on_validation_start(self, trainer, pl_module):
        # receive the global model and update the local model with global model
        # the 1st time validate() or train() is called.
        # expect user will validate the global model first (i.e. validate()), once that's done.
        # the metrics will be set.
        # The subsequent validate() calls will not trigger the receive update model.
        # Hence the validate() will be validating the local model.
        import time as _time
        print(f"[DIAG] on_validation_start called, metrics={self.metrics}, pl_module={pl_module is not None}", flush=True)
        if pl_module and self.metrics is None:
            _t0 = _time.monotonic()
            print("[DIAG] calling _receive_and_update_model...", flush=True)
            self._receive_and_update_model(trainer, pl_module)
            print(f"[DIAG] _receive_and_update_model done in {_time.monotonic()-_t0:.2f}s", flush=True)

    def on_validation_end(self, trainer, pl_module):
        if pl_module and self.metrics is None:
            self.metrics = _extract_metrics(trainer.callback_metrics)
            if self._is_evaluation:
                self._send_model(FLModel(metrics=self.metrics))
                self.reset_state(trainer)

    def _receive_and_update_model(self, trainer, pl_module):
        import time as _time
        _t0 = _time.monotonic()
        print("[DIAG] _receive_and_update_model: calling _receive_model...", flush=True)
        model = self._receive_model(trainer)
        print(f"[DIAG] _receive_model returned in {_time.monotonic()-_t0:.2f}s, model={model is not None}", flush=True)
        if model:
            if model.params:
                print(f"[DIAG] model.params has {len(model.params)} keys, calling load_state_dict...", flush=True)
                _t1 = _time.monotonic()
                try:
                    result = pl_module.load_state_dict(model.params, strict=self._load_state_dict_strict)
                    if result is not None:
                        missing_keys, unexpected_keys = result
                        if len(missing_keys) > 0:
                            self.logger.warning(
                                f"There were missing keys when loading the global state_dict: {missing_keys}"
                            )
                        if len(unexpected_keys) > 0:
                            self.logger.warning(
                                f"There were unexpected keys when loading the global state_dict: {unexpected_keys}"
                            )
                    print(f"[DIAG] load_state_dict done in {_time.monotonic()-_t1:.2f}s", flush=True)
                except Exception as e:
                    print(f"[DIAG] load_state_dict FAILED: {str(e)}", flush=True)
                    raise RuntimeError(f"Failed to load model state dict: {str(e)}")
            if model.current_round is not None:
                self.current_round = model.current_round
        print(f"[DIAG] _receive_and_update_model complete in {_time.monotonic()-_t0:.2f}s", flush=True)

    def _receive_model(self, trainer) -> FLModel:
        """Receives model from NVFlare."""
        import time as _time
        model = None
        _is_training = False
        _is_evaluation = False
        _is_submit_model = False
        print(f"[DIAG] _receive_model: rank={self.rank}", flush=True)
        if self.rank == 0:
            _t0 = _time.monotonic()
            print("[DIAG] _receive_model: calling receive()...", flush=True)
            model = receive()
            print(f"[DIAG] _receive_model: receive() done in {_time.monotonic()-_t0:.2f}s", flush=True)
            _is_training = is_train()
            _is_evaluation = is_evaluate()
            _is_submit_model = is_submit_model()
            print(f"[DIAG] _receive_model: train={_is_training}, eval={_is_evaluation}, submit={_is_submit_model}", flush=True)

        _t1 = _time.monotonic()
        print("[DIAG] _receive_model: calling strategy.broadcast(model)...", flush=True)
        model = trainer.strategy.broadcast(model, src=0)
        print(f"[DIAG] _receive_model: broadcast(model) done in {_time.monotonic()-_t1:.2f}s", flush=True)
        _t2 = _time.monotonic()
        self._is_training = trainer.strategy.broadcast(_is_training, src=0)
        self._is_evaluation = trainer.strategy.broadcast(_is_evaluation, src=0)
        self._is_submit_model = trainer.strategy.broadcast(_is_submit_model, src=0)
        print(f"[DIAG] _receive_model: broadcast(flags) done in {_time.monotonic()-_t2:.2f}s", flush=True)
        return model

    def _send_model(self, output_model: FLModel):
        try:
            send(output_model, clear_cache=False)
        except Exception as e:
            raise RuntimeError(f"failed to send FL model: {e}")


def _extract_metrics(metrics: Dict[str, Tensor]):
    result_metrics = {}
    for key, t in metrics.items():
        result_metrics[key] = t.item()
    return result_metrics
