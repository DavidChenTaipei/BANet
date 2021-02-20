banet = dict(
    lr_start = 0.000248, #0.049553127402395675, #5e-2,
    weight_decay= 5e-4,
    warmup_iters = 0,#3,
    start_epoch = 0,
    epoch = 360, ## 400 = 150000/(2975/8)
    #max_iter = 150000,
    #max_iter = 100000,
    im_root='../myBi/datasets/cityscapes',
    train_im_anns='../myBi/datasets/cityscapes/train.txt',
    val_im_anns='../myBi/datasets/cityscapes/val.txt',
    scales=[0.25, 2.],
    cropsize=[1024, 1024],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
