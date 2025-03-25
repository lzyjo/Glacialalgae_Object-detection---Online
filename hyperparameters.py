# Learning parameters


checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size (CHANGE ACCORDINGLY)
iterations =  1200 # number of iterations to train (CHANGE ACCORDINGLY)
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-5  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
checkpoint_freq = 120  # save checkpoint every __ iterations (CHANGE ACCORDINGLY)


# Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
# To convert iterations to epochs, divide iterations by the number of iterations per epoch
# The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
epochs = 300
decay_lr_at_epochs = [150, 250]

# epochs = iterations // (len(train_dataset) // 32)
# decay_lr_at_epochs = [it // (len(train_dataset) // 32) for it in decay_lr_at]


