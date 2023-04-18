"""
Contains RCE metrics.
"""
import copy
from functools import partial
from typing import Union

from tml.metrics import aggregation

import torch
import torchmetrics


def _smooth(
  value: torch.Tensor, label_smoothing: Union[float, torch.Tensor]
) -> Union[float, torch.Tensor]:
  """
  Smooth given values.
  Args:
    value: Value to smooth.
    label_smoothing: smoothing constant.
  Returns: Smoothed values.
  """
  return value * (1.0 - label_smoothing) + 0.5 * label_smoothing


def _binary_cross_entropy_with_clipping(
  predictions: torch.Tensor,
  target: torch.Tensor,
  epsilon: Union[float, torch.Tensor],
  reduction: str = "none",
) -> torch.Tensor:
  """
  Clip Predictions and apply binary cross entropy.
  This is done to match the implementation in keras at
  https://github.com/keras-team/keras/blob/r2.9/keras/backend.py#L5294-L5300
  Args:
    predictions: Predicted probabilities.
    target: Ground truth.
    epsilon: Epsilon fuzz factor used to clip the predictions.
    reduction: The reduction method to use.

  Returns: Binary cross entropy on the clipped predictions.

  """
  predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)
  bce = -target * torch.log(predictions + epsilon)
  bce -= (1.0 - target) * torch.log(1.0 - predictions + epsilon)
  if reduction == "mean":
    return torch.mean(bce)
  return bce


class RCE(torchmetrics.Metric):
  def __init__(
    self, from_logits: bool = False, label_smoothing: float = 0, epsilon: float = 1e-7, **kwargs
  ):
    """
    Args:
      from_logits: whether or not predictions are logits or probabilities.
      label_smoothing: label smoothing constant.
      epsilon: Epsilon fuzz factor used on the predictions probabilities when from_logits is False.
      **kwargs: Additional parameters supported by all torchmetrics.Metric.
    """
    super().__init__(**kwargs)
    self.from_logits = from_logits
    self.label_smoothing = label_smoothing
    self.epsilon = epsilon
    self.kwargs = kwargs

    self.mean_label = aggregation.StableMean(**kwargs)
    self.binary_cross_entropy = aggregation.StableMean(**kwargs)

    if self.from_logits:
      self.bce_loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    else:
      self.bce_loss_fn = partial(_binary_cross_entropy_with_clipping, epsilon=self.epsilon)

    # Used to compute non-accumulated batch metric if `forward` or `__call__` functions are used.
    self.batch_metric = copy.deepcopy(self)

  def update(
    self, predictions: torch.Tensor, target: torch.Tensor, weight: float = 1.0
  ) -> torch.Tensor:
    """
    Update the current rce.
    Args:
      predictions: Predicted values.
      target: Ground truth. Should have same shape as predictions.
      weight: The weight to use for the predicted values. Shape should be broadcastable to that of
       predictions.
    """
    target = _smooth(target, self.label_smoothing)
    self.mean_label.update(target, weight)
    self.binary_cross_entropy.update(
      self.bce_loss_fn(predictions, target, reduction="none"), weight
    )

  def compute(self) -> torch.Tensor:
    """
    Compute and return the accumulated rce.
    """
    baseline_mean = self.mean_label.compute()

    baseline_ce = _binary_cross_entropy_with_clipping(
      baseline_mean, baseline_mean, reduction="mean", epsilon=self.epsilon
    )

    pred_ce = self.binary_cross_entropy.compute()

    return (1.0 - (pred_ce / baseline_ce)) * 100

  def reset(self):
    """
    Reset the metric to its initial state.
    """
    super().reset()
    self.mean_label.reset()
    self.binary_cross_entropy.reset()

  def forward(self, *args, **kwargs):
    """
    Serves the dual purpose of both computing the metric on the current batch of inputs but also
        add the batch statistics to the overall accumulating metric state.
    Input arguments are the exact same as corresponding ``update`` method.
    The returned output is the exact same as the output of ``compute``.
    """
    self.update(*args, **kwargs)
    self.batch_metric.update(*args, **kwargs)
    batch_result = self.batch_metric.compute()
    self.batch_metric.reset()
    return batch_result


class NRCE(RCE):
  def __init__(
    self, from_logits: bool = False, label_smoothing: float = 0, epsilon: float = 1e-7, **kwargs
  ):
    """

    Args:
      from_logits: whether or not predictions are logits or probabilities.
      label_smoothing: label smoothing constant.
      epsilon: Epsilon fuzz factor used on the predictions probabilities when from_logits is False.
               It only used when computing the cross entropy but not when normalizing.
      **kwargs: Additional parameters supported by all torchmetrics.Metric.
    """
    super().__init__(from_logits=False, label_smoothing=0, epsilon=epsilon, **kwargs)
    self.nrce_from_logits = from_logits
    self.nrce_label_smoothing = label_smoothing
    self.mean_prediction = aggregation.StableMean()

    # Used to compute non-accumulated batch metric if `forward` or `__call__` functions are used.
    self.batch_metric = copy.deepcopy(self)

  def update(
    self,
    predictions: torch.Tensor,
    target: torch.Tensor,
    weight: Union[float, torch.Tensor] = 1.0,
  ):
    """
    Update the current nrce.
    Args:
      predictions: Predicted values.
      target: Ground truth. Should have same shape as predictions.
      weight: The weight to use for the predicted values. Shape should be broadcastable to that of
       predictions.
    """
    predictions = torch.sigmoid(predictions) if self.nrce_from_logits else predictions

    target = _smooth(target, self.nrce_label_smoothing)
    self.mean_label.update(target, weight)

    self.mean_prediction.update(predictions, weight)

    normalizer = self.mean_label.compute() / self.mean_prediction.compute()

    predictions = predictions * normalizer

    self.binary_cross_entropy.update(
      self.bce_loss_fn(predictions, target, reduction="none"), weight
    )

  def reset(self):
    """
    Reset the metric to its initial state.
    """
    super().reset()
    self.mean_prediction.reset()
