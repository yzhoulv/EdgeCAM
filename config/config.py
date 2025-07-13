class Config(object):
    env = 'default'
    num_classes = 2 # 52 # 659 # 8277 # 710 # 509 # 340
    loss = 'cross_entry'
    backbone = 'convnext_tiny'

    display = True
    finetune = False 

    train_root = '/data1/yangzhou/datasets/cls/Columbia'
    train_list = '/data1/yangzhou/datasets/cls/Columbia/Columbia.txt'
    val_root = '/data1/yangzhou/datasets/cls/Columbia'
    val_list = '/data1/yangzhou/datasets/cls/Columbia/Columbia.txt'


    checkpoints_path = './output/'
    save_interval = 1
    evl_interval = 1
    train_batch_size = 16  # batch size
    test_batch_size = 16
    input_shape = (3, 512, 512)
    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 4 # how many workers for loading data
    print_freq = 50  # print info every N batch

    max_epoch = 100
    lr = 1e-4  # initial learning rate
    lr_step = 100
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4
