from tensorflow.contrib.tpu.python.tpu.tpu_estimator import _SignalsHelper


def _signals_helper___init__(self, signals):
    self._signal_keys = []
    for key in sorted(signals.keys()):
      self._signal_keys.append(key)

def _signals_helper_as_tensor_list(signals):
    return [signals[key] for key in sorted(signals.keys())]

_SignalsHelper.__init__ = _signals_helper___init__
_SignalsHelper.as_tensor_list = _signals_helper_as_tensor_list
