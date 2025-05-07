checkpoint=None
batch_size=8
iterations=1200
workers=4
print_freq=200
lr=1e-5
decay_lr_at=[80000, 100000]
decay_lr_to=0.1
momentum=0.9
weight_decay=5e-4
grad_clip=None
checkpoint_freq=120
epoch_num=1000
decay_lr_at_epochs = [500, 800]


"""params_dict = {
    "checkpoint": checkpoint,
    "batch_size": batch_size,
    "iterations": iterations,
    "workers": workers,
    "print_freq": print_freq,
    "lr": lr,
    "decay_lr_at": decay_lr_at,
    "decay_lr_to": decay_lr_to,
    "momentum": momentum,
    "weight_decay": weight_decay,
    "grad_clip": grad_clip,
    "checkpoint_freq": checkpoint_freq,
    "Epoch_num": epoch_num,
    "decay_lr_at_epochs": decay_lr_at_epochs
}"""

"""# Change: Learning rate
params_1 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            lr=1e-6)  # Learning rate for the optimizer

# Change: Number of iterations
params_2 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            iterations=1500)

# Change: Number of iterations
params_3 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            iterations=2000)

# Change: Batch size
params_4 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            batch_size=4)

# Change: Decay learning rate at different iterations
params_5 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            decay_lr_at=[600, 1000])

# Change: Decay learning rate at different iterations
params_6 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            decay_lr_at=[250, 400])

# Change: Learning rate
params_7 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            lr=3e-5)

# Change: Number of iterations
params_8 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            iterations=1000)

# Change: Batch size
params_9 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                            batch_size=20)

# Change: Decay learning rate at different iterations
params_10 = Hyperparameters(checkpoint=None,  # Path to the checkpoint file (if any)
                             decay_lr_at=[150, 250])

params_dict = vars(params)  # Convert params to a dictionary for easy access
# This is then imported into train.py/utils.py
# and used to set the hyperparameters for training/training output fil


"""

"""class Hyperparameters:

    def __init__(self,
                 checkpoint=None,
                 batch_size=8,
                 iterations=1200,
                 workers=4,
                 print_freq=200,
                 lr=1e-5,
                 decay_lr_at=[80000, 100000],
                 decay_lr_to=0.1,
                 momentum=0.9,
                 weight_decay=5e-4,
                 grad_clip=None,
                 checkpoint_freq=120,
                 epochs=1000,
                 decay_lr_at_epochs=[300, 600]):
        
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.iterations = iterations
        self.workers = workers
        self.print_freq = print_freq
        self.lr = lr
        self.decay_lr_at = decay_lr_at
        self.decay_lr_to = decay_lr_to
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.checkpoint_freq = checkpoint_freq
        self.epochs = epochs
        self.decay_lr_at_epochs = decay_lr_at_epochs



    def modify_param(self, attribute, direction, rate):
        rates = {
            "safe": 1.1,
            "normal": 1.25,
            "aggressive": 1.5
        }
        if attribute not in vars(self):
            raise ValueError(f"Invalid attribute. Choose from: {list(vars(self).keys())}")
        if direction not in ["increase", "decrease"]:
            raise ValueError("Direction must be 'increase' or 'decrease'.")
        if rate not in rates:
            raise ValueError(f"Rate must be one of {list(rates.keys())}.")

        factor = rates[rate]
        if direction == "decrease":
            factor = 1 / factor

        current_value = getattr(self, attribute)
        if isinstance(current_value, (int, float)):
            setattr(self, attribute, current_value * factor)
        else:
            raise TypeError("Only numeric attributes can be modified.")

        return self"""

