import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=1e-3, min_lr=1e-5, num_cycles=0.5, last_epoch=-1):
        """
        参数:
            optimizer: 优化器
            warmup_steps: warmup 步数
            total_steps: 总训练步数
            base_lr: 最大学习率
            min_lr: 最小学习率
            num_cycles: cosine周期数（0.5 表示半个周期，不重启；1表示完整周期；>1 表示多周期重启）
            last_epoch: 恢复用
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_cycles = num_cycles

        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = base_lr

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * self.num_cycles * progress))
                return cosine_decay * (1 - min_lr / base_lr) + (min_lr / base_lr)

        self.scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
