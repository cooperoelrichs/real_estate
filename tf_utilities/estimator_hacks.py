import os
import copy

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.eager import context
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.training import evaluation
from tensorflow.python.training import saver
from tensorflow.python.util import compat_internal

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_context

from tensorflow.python.estimator.estimator import (
    Estimator, _get_replica_device_setter, _verify_model_fn_args,
    warm_starting_util, _extract_metric_update_ops, _write_dict_to_summary)
from tensorflow.contrib.tpu.python.tpu.tpu_estimator import TPUEstimator


class TensorBoardCompatibleEvaluator():
    def evaluation_directory(self, name):
        '''
        Place the 'eval' directory next to the model directory rather than
        inside of it.
        '''
        if name is None:
            folder_name = 'eval'
        else:
            folder_name = 'eval_' + name

        return os.path.join(self._model_dir, '..', folder_name)

    def _evaluate_model(
        self, input_fn, hooks=None, checkpoint_path=None, name=''
    ):
        """Evaluates the model using the training.evaluation library."""
        # Check that model has been trained (if nothing has been set explicitly).
        if not checkpoint_path:
          latest_path = saver.latest_checkpoint(self._model_dir)
          if not latest_path:
            raise ValueError('Could not find trained model in model_dir: {}.'.
                             format(self._model_dir))
          checkpoint_path = latest_path

        # <modifications>
        # Setup output directory.
        # eval_dir = os.path.join(self._model_dir, 'eval' if not name else
        #                         'eval_' + name)
        eval_dir = TensorBoardCompatibleEvaluator.evaluation_directory(self, name)
        # </modifications>

        with ops.Graph().as_default() as g:
          random_seed.set_random_seed(self._config.tf_random_seed)
          global_step_tensor = self._create_and_assert_global_step(g)
          features, labels, input_hooks = (
              self._get_features_and_labels_from_input_fn(
                  input_fn, model_fn_lib.ModeKeys.EVAL))
          estimator_spec = self._call_model_fn(
              features, labels, model_fn_lib.ModeKeys.EVAL, self.config)

          if model_fn_lib.LOSS_METRIC_KEY in estimator_spec.eval_metric_ops:
            raise ValueError(
                'Metric with name "%s" is not allowed, because Estimator ' % (
                    model_fn_lib.LOSS_METRIC_KEY) +
                'already defines a default metric with the same name.')
          estimator_spec.eval_metric_ops[
              model_fn_lib.LOSS_METRIC_KEY] = metrics_lib.mean(estimator_spec.loss)

          update_op, eval_dict = _extract_metric_update_ops(
              estimator_spec.eval_metric_ops)

          if ops.GraphKeys.GLOBAL_STEP in eval_dict:
            raise ValueError(
                'Metric with name `global_step` is not allowed, because Estimator '
                'already defines a default metric with the same name.')
          eval_dict[ops.GraphKeys.GLOBAL_STEP] = global_step_tensor

          all_hooks = list(input_hooks)
          all_hooks.extend(hooks)
          all_hooks.extend(list(estimator_spec.evaluation_hooks or []))

          eval_results = evaluation._evaluate_once(  # pylint: disable=protected-access
              checkpoint_path=checkpoint_path,
              master=self._config.evaluation_master,
              scaffold=estimator_spec.scaffold,
              eval_ops=update_op,
              final_ops=eval_dict,
              hooks=all_hooks,
              config=self._session_config)

          _write_dict_to_summary(
              output_dir=eval_dir,
              dictionary=eval_results,
              current_global_step=eval_results[ops.GraphKeys.GLOBAL_STEP])

        return eval_results


class EstimatorWithTensorBoardCompatibility(Estimator):
    def __init__(
        self, model_fn, model_dir=None, config=None, params=None,
        warm_start_from=None
    ):
        if context.executing_eagerly():
          raise RuntimeError(
              'Estimators are not supported when eager execution is enabled.')

        if context.executing_eagerly():
          raise RuntimeError(
              'Estimators are not supported when eager execution is enabled.')

        # <modifications>
        # This check has to be removed.
        # Estimator._assert_members_are_not_overridden(self)
        # </modifications>

        if config is None:
          self._config = run_config.RunConfig()
          logging.info('Using default config.')
        else:
          if not isinstance(config, run_config.RunConfig):
            raise ValueError(
                'config must be an instance of RunConfig, but provided %s.' %
                config)
          self._config = config

        # Model directory.
        model_dir = compat_internal.path_to_str(model_dir)
        if (model_dir is not None) and (self._config.model_dir is not None):
          if model_dir != self._config.model_dir:
            # TODO(alanyee): remove this suppression after it is no longer needed
            # pylint: disable=g-doc-exception
            raise ValueError(
                "model_dir are set both in constructor and RunConfig, but with "
                "different values. In constructor: '{}', in RunConfig: "
                "'{}' ".format(model_dir, self._config.model_dir))
            # pylint: enable=g-doc-exception

        self._model_dir = model_dir or self._config.model_dir
        if self._model_dir is None:
          self._model_dir = tempfile.mkdtemp()
          logging.warning('Using temporary folder as model directory: %s',
                          self._model_dir)
        if self._config.model_dir is None:
          self._config = self._config.replace(model_dir=self._model_dir)
        logging.info('Using config: %s', str(vars(self._config)))

        if self._config.session_config is None:
          self._session_config = config_pb2.ConfigProto(allow_soft_placement=True)
        else:
          self._session_config = self._config.session_config

        self._device_fn = _get_replica_device_setter(self._config)

        if model_fn is None:
          raise ValueError('model_fn must be provided to Estimator.')
        _verify_model_fn_args(model_fn, params)
        self._model_fn = model_fn
        self._params = copy.deepcopy(params or {})

        # pylint: disable=protected-access
        self._warm_start_settings = (
            warm_starting_util._get_default_warm_start_settings(warm_start_from))
        # pylint: enable=protected-access

    def _evaluate_model(
        self, input_fn, hooks=None, checkpoint_path=None, name=''
    ):
        return TensorBoardCompatibleEvaluator._evaluate_model(
            self, input_fn, hooks, checkpoint_path, name
        )


class TPUEstimatorWithTensorBoardCompatibility(TPUEstimator):
    def __init__(self,
        model_fn=None,
        model_dir=None,
        config=None,
        params=None,
        use_tpu=True,
        train_batch_size=None,
        eval_batch_size=None,
        predict_batch_size=None,
        batch_axis=None
    ):

        if config is None or not isinstance(config, tpu_config.RunConfig):
          raise ValueError(
              '`config` must be provided with type `tpu_config.RunConfig`')

        if params is not None and any(k in params for k in _RESERVED_PARAMS_KEYS):
          raise ValueError('{} are reserved keys but existed in params {}.'.format(
              _RESERVED_PARAMS_KEYS, params))

        if use_tpu:
          # Perform some very basic validations. More validations will be found in
          # _TPUContext.
          if train_batch_size is None:
            raise ValueError('`train_batch_size` cannot be `None`')
          util_lib.check_positive_integer(train_batch_size, 'train_batch_size')

          if (not config.tpu_config.per_host_input_for_training and
              config.tpu_config.computation_shape):
            raise ValueError(
                'Model parallelism only supports per host input for training. '
                'Please adjust TPURunconfig.per_host_input_for_training.')

          if eval_batch_size is not None:
            util_lib.check_positive_integer(eval_batch_size, 'eval_batch_size')

          if predict_batch_size is not None:
            util_lib.check_positive_integer(predict_batch_size,
                                            'predict_batch_size')

        # Verifies the model_fn signature according to Estimator framework.
        estimator_lib._verify_model_fn_args(model_fn, params)  # pylint: disable=protected-access
        # We cannot store config and params in this constructor as parent
        # constructor might change them, such as assigning a temp dir for
        # config.model_dir.
        model_function = self._augment_model_fn(model_fn, batch_axis)

        # Passing non-None params as wrapped model_fn has it.
        params = params or {}

        # <modifications>
        # super(TPUEstimator, self).__init__(
        #     model_fn=model_function,
        #     model_dir=model_dir,
        #     config=config,
        #     params=params)
        EstimatorWithTensorBoardCompatibility.__init__(
            self,
            model_fn=model_function,
            model_dir=model_dir,
            config=config,
            params=params)
        # </modifications>

        self._iterations_per_training_loop = (
            self._config.tpu_config.iterations_per_loop)

        # All properties passed to _TPUContext are immutable.
        # pylint: disable=protected-access
        self._ctx = tpu_context._get_tpu_context(
            self._config, train_batch_size,
            eval_batch_size, predict_batch_size,
            use_tpu)



    def _evaluate_model(
        self, input_fn, hooks=None, checkpoint_path=None, name=''
    ):
        return TensorBoardCompatibleEvaluator._evaluate_model(
            self, input_fn, hooks, checkpoint_path, name
        )
