from .hookbase import HookBase


class DeepSpeedHook(HookBase):
    def after_iter(self) -> None:
        self.trainer.model.backward(self.trainer.loss_dict["total_loss"])
        for name, param in self.trainer.model.named_parameters():
            if param.grad is not None:
                print(name, param.grad.norm().item())
        self.trainer._call_hooks("after_backward")
        self.trainer.model.step()
        self.trainer._call_hooks("after_step")

        if self.trainer._clip_grad_norm is not None and self.trainer._clip_grad_norm > 0.0:
            self.trainer.log(
                self.trainer.cur_iter, smooth=False, grad_norm=self.trainer.optimizer._global_grad_norm
            )
