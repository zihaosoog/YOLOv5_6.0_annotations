# Trick and tips

## Build env for pytorch (Linux)
Step:
> Gpu driver -> Anaconda -> CUDA 11.4 -> CUDNN 8.2.4 -> pytorch (CUDA 11.3)

## Pytorch Parallel load
- [ ] DDP and loaddata
[(here)](https://www.cnblogs.com/rossiXYZ/p/15496268.html)

## Train model trick

### 自动混合精度训练
    scaler = amp.GradScaler(enabled=cuda)
### EarlyStop method
防止模型继续训练而导致的性能下降 [(here)](https://blog.csdn.net/zwqjoy/article/details/86677030)

    stopper = EarlyStopping(patience=opt.patience)
### BN and Dropout setting
BN for trian mode use the mean and variance of this batch, but it use mean and variance of all data in val mode.
Dropout is True in train mode, but not in val.

    model.train()

### Multi-scale train
- [ ] how to do

### Callback
- [ ] what means

### EMA
- [ ] how to do


