def lr_lambda(epoch):
    warm_up_epoch = 20  
    lr_max = 0.01
    lr_min = 3e-4
    if epoch <= warm_up_epoch:
        lr = max((epoch / warm_up_epoch) * lr_max, lr_min)
    else:
        lr = max(lr_min, lr_max * 0.9**(epoch - warm_up_epoch))
    lr_rate = lr / lr_min
    return lr_rate

