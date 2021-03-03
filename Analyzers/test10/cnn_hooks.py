import tensorflow as tf

# ______________________________________________________________________________
# from tensorflow/models: official/utils/logs/hooks.py

class ExamplesPerSecondHook(tf.train.SessionRunHook):
  """Hook to print out examples per second.

  Total time is tracked and then divided by the total number of steps
  to get the average step time and then batch_size is used to determine
  the running average of examples per second. The examples per second for the
  most recent interval is also logged.
  """

  def __init__(self,
               batch_size,
               every_n_steps=None,
               every_n_secs=None,
               warm_steps=0,
               metric_logger=None):
    """Initializer for ExamplesPerSecondHook.

    Args:
      batch_size: Total batch size across all workers used to calculate
        examples/second from global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds. Exactly one of the
        `every_n_steps` or `every_n_secs` should be set.
      warm_steps: The number of steps to be skipped before logging and running
        average calculation. warm_steps steps refers to global steps across all
        workers, not on each worker
      metric_logger: instance of `BenchmarkLogger`, the benchmark logger that
          hook should use to write the log. If None, BaseBenchmarkLogger will
          be used.

    Raises:
      ValueError: if neither `every_n_steps` or `every_n_secs` is set, or
      both are set.
    """

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps"
                       " and every_n_secs should be provided.")

    self._logger = metric_logger or logger.BaseBenchmarkLogger()

    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size
    self._warm_steps = warm_steps

  def begin(self):
    """Called once before using the session to check global step."""
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use StepCounterHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    """Called before each call to run().

    Args:
      run_context: A SessionRunContext object.

    Returns:
      A SessionRunArgs object or None if never triggered.
    """
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    """Called after each call to run().

    Args:
      run_context: A SessionRunContext object.
      run_values: A SessionRunValues object.
    """
    global_step = run_values.results

    if self._timer.should_trigger_for_step(
        global_step) and global_step > self._warm_steps:
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        # average examples per second is based on the total (accumulative)
        # training steps and training time so far
        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        # current examples per second is based on the elapsed training steps
        # and training time per batch
        current_examples_per_sec = self._batch_size * (
            elapsed_steps / elapsed_time)

        self._logger.log_metric(
            "average_examples_per_sec", average_examples_per_sec,
            global_step=global_step)

        self._logger.log_metric(
            "current_examples_per_sec", current_examples_per_sec,
            global_step=global_step)


# ______________________________________________________________________________
# from tensorflow/models: official/utils/logs/metric_hook.py

class LoggingMetricHook(tf.train.LoggingTensorHook):
  """Hook to log benchmark metric information.

  This hook is very similar as tf.train.LoggingTensorHook, which logs given
  tensors every N local steps, every N seconds, or at the end. The metric
  information will be logged to given log_dir or via metric_logger in JSON
  format, which can be consumed by data analysis pipeline later.

  Note that if `at_end` is True, `tensors` should not include any tensor
  whose evaluation produces a side effect such as consuming additional inputs.
  """

  def __init__(self, tensors, metric_logger=None,
               every_n_iter=None, every_n_secs=None, at_end=False):
    """Initializer for LoggingMetricHook.

    Args:
      tensors: `dict` that maps string-valued tags to tensors/tensor names,
          or `iterable` of tensors/tensor names.
      metric_logger: instance of `BenchmarkLogger`, the benchmark logger that
          hook should use to write the log.
      every_n_iter: `int`, print the values of `tensors` once every N local
          steps taken on the current worker.
      every_n_secs: `int` or `float`, print the values of `tensors` once every N
          seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
          provided.
      at_end: `bool` specifying whether to print the values of `tensors` at the
          end of the run.

    Raises:
      ValueError:
        1. `every_n_iter` is non-positive, or
        2. Exactly one of every_n_iter and every_n_secs should be provided.
        3. Exactly one of log_dir and metric_logger should be provided.
    """
    super(LoggingMetricHook, self).__init__(
        tensors=tensors,
        every_n_iter=every_n_iter,
        every_n_secs=every_n_secs,
        at_end=at_end)

    if metric_logger is None:
      raise ValueError("metric_logger should be provided.")
    self._logger = metric_logger

  def begin(self):
    super(LoggingMetricHook, self).begin()
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use LoggingMetricHook.")
    if self._global_step_tensor.name not in self._current_tensors:
      self._current_tensors[self._global_step_tensor.name] = (
          self._global_step_tensor)

  def after_run(self, unused_run_context, run_values):
    # should_trigger is a internal state that populated at before_run, and it is
    # using self_timer to determine whether it should trigger.
    if self._should_trigger:
      self._log_metric(run_values.results)

    self._iter_count += 1

  def end(self, session):
    if self._log_at_end:
      values = session.run(self._current_tensors)
      self._log_metric(values)

  def _log_metric(self, tensor_values):
    self._timer.update_last_triggered_step(self._iter_count)
    global_step = tensor_values[self._global_step_tensor.name]
    # self._tag_order is populated during the init of LoggingTensorHook
    for tag in self._tag_order:
      self._logger.log_metric(tag, tensor_values[tag], global_step=global_step)
