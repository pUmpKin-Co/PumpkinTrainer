import logging

from .hookbase import HookBase

logger = logging.getLogger("train")


class EvalHook(HookBase):
    """Run an evaluation function periodically.

    It is executed every ``period`` epochs and after the last epoch.
    """

    def __init__(self, period: int, task: str = "cls"):
        """
        Args:
            period (int): The period to run ``eval_func``. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_func (callable): A function which takes no arguments, and
                returns a dict of evaluation metrics.
        """
        super(EvalHook, self).__init__()
        self._period = period

    def _eval_func(self):
        raise NotImplementedError("Please implement the _eval_func method")


class EpochEvalHook(EvalHook):
    def after_epoch(self):
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._eval_func()


class IterEvalHook(EvalHook):
    def after_iter(self):
        if self.every_n_iters(self._period) or self.is_last_iter():
            self._eval_func()
