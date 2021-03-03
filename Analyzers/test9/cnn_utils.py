import tensorflow as tf

# ______________________________________________________________________________
# from tensorflow/models: official/utils/flags/core.py

flags = tf.flags

def register_key_flags_in_core(f):
  """Defines a function in core.py, and registers its key flags.

  absl uses the location of a flags.declare_key_flag() to determine the context
  in which a flag is key. By making all declares in core, this allows model
  main functions to call flags.adopt_module_key_flags() on core and correctly
  chain key flags.

  Args:
    f:  The function to be wrapped

  Returns:
    The "core-defined" version of the input function.
  """

  def core_fn(*args, **kwargs):
    key_flags = f(*args, **kwargs)
    [flags.declare_key_flag(fl) for fl in key_flags]  # pylint: disable=expression-not-assigned
  return core_fn


@register_key_flags_in_core
def define_reiam_base_flags():
  """Register base flags.
  Args:
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    num_epochs: Create a flag to specify the number of training epochs.
    epochs_between_evals: Create a flag to specify the frequency of testing.
    stop_threshold: Create a flag to specify a threshold accuracy or other
      eval metric which should trigger the end of training.
    batch_size: Create a flag to specify the batch size.
    num_gpu: Create a flag to specify the number of GPUs used.
    hooks: Create a flag to specify hooks for logging.
    export_dir: Create a flag to specify where a SavedModel should be exported.
  Returns:
    A list of flags for core.py to marks as key flags.
  """

  import functools
  help_wrap = functools.partial(flags.text_wrap, length=80, indent="",
                                firstline_indent="\n")

  key_flags = []

  flags.DEFINE_string(
      name="data_dir", short_name="dd", default="/tmp",
      help=help_wrap("The location of the input data."))
  key_flags.append("data_dir")

  flags.DEFINE_string(
      name="model_dir", short_name="md", default="/tmp",
      help=help_wrap("The location of the model checkpoint files."))
  key_flags.append("model_dir")

  flags.DEFINE_boolean(
      name="clean", default=False,
      help=help_wrap("If set, model_dir will be removed if it exists."))
  key_flags.append("clean")

  flags.DEFINE_integer(
      name="num_epochs", short_name="te", default=1,
      help=help_wrap("The number of epochs used to train."))
  key_flags.append("num_epochs")

  flags.DEFINE_integer(
      name="epochs_between_evals", short_name="ebe", default=1,
      help=help_wrap("The number of training epochs to run between "
                     "evaluations."))
  key_flags.append("epochs_between_evals")

  flags.DEFINE_float(
      name="stop_threshold", short_name="st", default=None,
      help=help_wrap("If passed, training will stop at the earlier of "
                     "num_epochs and when the evaluation metric is  "
                     "greater than or equal to stop_threshold."))
  key_flags.append("stop_threshold")

  flags.DEFINE_integer(
      name="batch_size", short_name="bs", default=32,
      help=help_wrap("Batch size for training and evaluation. When using "
                     "multiple gpus, this is the global batch size for "
                     "all devices. For example, if the batch size is 32 "
                     "and there are 4 GPUs, each GPU will get 8 examples on "
                     "each step."))
  key_flags.append("batch_size")

  flags.DEFINE_integer(
      name="num_gpus", short_name="ng",
      default=1 if tf.test.is_gpu_available() else 0,
      help=help_wrap(
          "How many GPUs to use with the DistributionStrategies API. The "
          "default is 1 if TensorFlow can detect a GPU, and 0 otherwise."))
  key_flags.append("num_gpus")

  # Construct a pretty summary of hooks.
  hook_list_str = (
      u"\ufeff  Hook:\n" + u"\n".join([u"\ufeff    {}".format(key) for key
                                       in HOOKS]))
  flags.DEFINE_list(
      name="hooks", short_name="hk", default="LoggingTensorHook",
      help=help_wrap(
          u"A list of (case insensitive) strings to specify the names of "
          u"training hooks.\n{}\n\ufeff  Example: `--hooks ProfilerHook,"
          u"ExamplesPerSecondHook`\n See official.utils.logs.hooks_helper "
          u"for details.".format(hook_list_str)))
  key_flags.append("hooks")

  flags.DEFINE_string(
      name="export_dir", short_name="ed", default=None,
      help=help_wrap("If set, a SavedModel serialization of the model will "
                     "be exported to this directory at the end of training. "
                     "See the README for more details and relevant links."))
  key_flags.append("export_dir")

  flags.DEFINE_enum(
      name="data_format", short_name="df", default="channels_last",
      enum_values=["channels_first", "channels_last"],
      help=help_wrap(
            "A flag to override the data format used in the model. "
            "channels_first provides a performance boost on GPU but is not "
            "always compatible with CPU. If left unspecified, the data format "
            "will be chosen automatically based on whether TensorFlow was "
            "built for CPU or GPU."))
  key_flags.append("data_format")

  flags.DEFINE_boolean(
      name="shuffle", short_name="sf", default=True,
      help=help_wrap("Enable use of shuffled datasets for input pipeline."))
  key_flags.append("shuffle")

  flags.DEFINE_integer(
      name="shuffle_buffer_size", short_name="sbs", default=10000,
      help=help_wrap("Buffer size to use for shuffling. "
                     "A large buffer size ensures better shuffling, but "
                     "increases memory usage and startup time."))
  key_flags.append("shuffle_buffer_size")

  flags.DEFINE_boolean(
      name="prefetch", short_name="pf", default=True,
      help=help_wrap("Enable use of prefetched datasets for input pipeline."))
  key_flags.append("prefetch")

  flags.DEFINE_integer(
      name="prefetch_buffer_size", short_name="pbs", default=1,
      help=help_wrap("An int specifying the number of feature batches to prefetch "
                     "for performance improvement. Recommended value is the number "
                     "of batches consumed per training step."))
  key_flags.append("prefetch_buffer_size")

  tf.app.flags.DEFINE_boolean(
      name="log_device_placement", default=False,
      help=help_wrap("Whether to log device placement."))
  key_flags.append("log_device_placement")

  flags.DEFINE_integer(
      name="num_intra_threads", default=4,
      help=help_wrap("Number of threads to use for intra-op parallelism. If "
                     "set to 0, the system will pick an appropriate number."))
  key_flags.append("num_intra_threads")

  flags.DEFINE_integer(
      name="num_inter_threads", default=0,
      help=help_wrap("Number of threads to use for inter-op parallelism. If "
                     "set to 0, the system will pick an appropriate number."))
  key_flags.append("num_inter_threads")

  flags.DEFINE_integer(
      name="tf_random_seed", default=2023,
      help=help_wrap("The TensorFlow random seed. Useful for debugging NaNs, "
                     "as this can be set to various values to see if the NaNs "
                     "depend on the seed."))
  key_flags.append("tf_random_seed")

  flags.DEFINE_enum(
      name="benchmark_logger_type", default="BaseBenchmarkLogger",
      enum_values=["BaseBenchmarkLogger", "BenchmarkFileLogger",
                   "BenchmarkBigQueryLogger"],
      help=help_wrap("The type of benchmark logger to use. Defaults to using "
                     "BaseBenchmarkLogger which logs to STDOUT. Different "
                     "loggers will require other flags to be able to work."))
  key_flags.append("benchmark_logger_type")

  flags.DEFINE_string(
      name="benchmark_test_id", short_name="bti", default=None,
      help=help_wrap("The unique test ID of the benchmark run. It could be the "
                     "combination of key parameters. It is hardware "
                     "independent and could be used compare the performance "
                     "between different test runs. This flag is designed for "
                     "human consumption, and does not have any impact within "
                     "the system."))
  key_flags.append("benchmark_test_id")

  flags.DEFINE_string(
      name="benchmark_log_dir", short_name="bld", default=None,
      help=help_wrap("The location of the benchmark logging."))
  key_flags.append("benchmark_log_dir")

  return key_flags

def clear_flags():
  for name in list(flags.FLAGS):
    delattr(flags.FLAGS, name)

def unparse_flags():
  flags.FLAGS.unparse_flags()

def set_defaults(**kwargs):
  #flags.FLAGS.remove_flag_values(kwargs.keys())
  for key, value in kwargs.iteritems():
    flags.FLAGS.set_default(name=key, value=value)


# ______________________________________________________________________________
# from tensorflow/models: official/utils/logs/hooks_helper.py

from cnn_hooks import ExamplesPerSecondHook, LoggingMetricHook
from cnn_benchmark import get_benchmark_logger

def get_train_hooks(name_list, use_tpu=False, **kwargs):
  """Factory for getting a list of TensorFlow hooks for training by name.

  Args:
    name_list: a list of strings to name desired hook classes. Allowed:
      LoggingTensorHook, ProfilerHook, ExamplesPerSecondHook, which are defined
      as keys in HOOKS
    use_tpu: Boolean of whether computation occurs on a TPU. This will disable
      hooks altogether.
    **kwargs: a dictionary of arguments to the hooks.

  Returns:
    list of instantiated hooks, ready to be used in a classifier.train call.

  Raises:
    ValueError: if an unrecognized name is passed.
  """

  if not name_list:
    return []

  if use_tpu:
    tf.logging.warning("hooks_helper received name_list `{}`, but a TPU is "
                       "specified. No hooks will be used.".format(name_list))
    return []

  train_hooks = []
  for name in name_list:
    hook_name = HOOKS.get(name.strip().lower())
    if hook_name is None:
      raise ValueError('Unrecognized training hook requested: {}'.format(name))
    else:
      train_hooks.append(hook_name(**kwargs))

  return train_hooks


def get_logging_tensor_hook(every_n_iter=100, tensors_to_log=None, **kwargs):  # pylint: disable=unused-argument
  """Function to get LoggingTensorHook.

  Args:
    every_n_iter: `int`, print the values of `tensors` once every N local
      steps taken on the current worker.
    tensors_to_log: List of tensor names or dictionary mapping labels to tensor
      names. If not set, log _TENSORS_TO_LOG by default.
    **kwargs: a dictionary of arguments to LoggingTensorHook.

  Returns:
    Returns a LoggingTensorHook with a standard set of tensors that will be
    printed to stdout.
  """
  if tensors_to_log is None:
    tensors_to_log = _TENSORS_TO_LOG

  return tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=every_n_iter)


def get_profiler_hook(model_dir, save_steps=1000, **kwargs):  # pylint: disable=unused-argument
  """Function to get ProfilerHook.

  Args:
    model_dir: The directory to save the profile traces to.
    save_steps: `int`, print profile traces every N steps.
    **kwargs: a dictionary of arguments to ProfilerHook.

  Returns:
    Returns a ProfilerHook that writes out timelines that can be loaded into
    profiling tools like chrome://tracing.
  """
  return tf.train.ProfilerHook(
      save_steps=save_steps, output_dir=model_dir,
      show_dataflow=True, show_memory=True)


def get_examples_per_second_hook(every_n_secs=600,
                                 batch_size=128,
                                 warm_steps=5,
                                 **kwargs):  # pylint: disable=unused-argument
  """Function to get ExamplesPerSecondHook.

  Args:
    every_n_steps: `int`, print current and average examples per second every
      N steps.
    batch_size: `int`, total batch size used to calculate examples/second from
      global time.
    warm_steps: skip this number of steps before logging and running average.
    **kwargs: a dictionary of arguments to ExamplesPerSecondHook.

  Returns:
    Returns a ProfilerHook that writes out timelines that can be loaded into
    profiling tools like chrome://tracing.
  """
  return ExamplesPerSecondHook(
      batch_size=batch_size, every_n_secs=every_n_secs,
      warm_steps=warm_steps, metric_logger=get_benchmark_logger())


def get_logging_metric_hook(tensors_to_log=None,
                            every_n_secs=600,
                            **kwargs):  # pylint: disable=unused-argument
  """Function to get LoggingMetricHook.

  Args:
    tensors_to_log: List of tensor names or dictionary mapping labels to tensor
      names. If not set, log _TENSORS_TO_LOG by default.
    every_n_secs: `int`, the frequency for logging the metric. Default to every
      10 mins.

  Returns:
    Returns a LoggingMetricHook that saves tensor values in a JSON format.
  """
  if tensors_to_log is None:
    tensors_to_log = _TENSORS_TO_LOG
  return LoggingMetricHook(
      tensors=tensors_to_log,
      metric_logger=get_benchmark_logger(),
      every_n_secs=every_n_secs)


def get_nan_tensor_hook(loss_tensor, fail_on_nan_loss=True, **kwargs):  # pylint: disable=unused-argument
  """Function to get NanTensorHook.

  Args:
    loss_tensor: `Tensor`, the loss tensor.
    fail_on_nan_loss: `bool`, whether to raise exception when loss is NaN.

  Returns:
    Returns a NanTensorHook that monitors the loss tensor and stops training
    if loss is NaN.
  """
  nan_tensor_hook = tf.train.NanTensorHook(
      loss_tensor=loss_tensor,
      fail_on_nan_loss=fail_on_nan_loss)

  from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
  def begin(self):
    self._loss_tensor = _as_graph_element('cross_entropy')
  nan_tensor_hook.begin = begin
  return nan_tensor_hook


_TENSORS_TO_LOG = {
    'learning_rate': 'learning_rate',
    'cross_entropy': 'cross_entropy',
    'train_accuracy': 'train_accuracy',
    'train_accuracy_at_k': 'train_accuracy_at_k',
}

# A dictionary to map one hook name and its corresponding function
HOOKS = {
    'loggingtensorhook': get_logging_tensor_hook,
    'profilerhook': get_profiler_hook,
    'examplespersecondhook': get_examples_per_second_hook,
    'loggingmetrichook': get_logging_metric_hook,
    'nantensorhook': get_nan_tensor_hook,
}


# ______________________________________________________________________________
# from tensorflow/models: official/utils/misc/model_helpers.py

def past_stop_threshold(stop_threshold, eval_metric):
  """Return a boolean representing whether a model should be stopped.

  Args:
    stop_threshold: float, the threshold above which a model should stop
      training.
    eval_metric: float, the current value of the relevant metric to check.

  Returns:
    True if training should stop, False otherwise.

  Raises:
    ValueError: if either stop_threshold or eval_metric is not a number
  """
  if stop_threshold is None:
    return False

  if not isinstance(stop_threshold, numbers.Number):
    raise ValueError("Threshold for checking stop conditions must be a number.")
  if not isinstance(eval_metric, numbers.Number):
    raise ValueError("Eval metric being checked against stop conditions "
                     "must be a number.")

  if eval_metric >= stop_threshold:
    tf.logging.info(
        "Stop threshold of {} was passed with metric value {}.".format(
            stop_threshold, eval_metric))
    return True

  return False


def apply_clean(flags_obj):
  if flags_obj.clean and tf.gfile.Exists(flags_obj.model_dir):
    tf.logging.info("--clean flag set. Removing existing model dir: {}".format(
        flags_obj.model_dir))
    tf.gfile.DeleteRecursively(flags_obj.model_dir)

