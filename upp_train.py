import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.unetpp.unet2plus import UNet_2Plus
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine import *
import os
import sys

from utils import *
from configs.config_upp_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
parser.add_argument('--wandb', action='store_true', help='whether wandb or not')
parser.add_argument('--epochs', action='store_true', help='whether epochs or not')
parser.add_argument('--en', type=int, default=10, help="rounds of training")
args = parser.parse_args()

def main(config):
    if args.epochs:
        config.epochs = args.en
    # https://wandb.ai/
    if args.wandb:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="koala",

            # track hyperparameters and run metadata
            config={
            "learning_rate": config.lr,
            "architecture": "UNETPP",
            "dataset": "ISIC18",
            "epochs": config.epochs,
            }
        )
    else :
        wandb = None

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'unetpp':
        model = UNet_2Plus(
            in_channels=model_cfg['input_channels'],
            n_classes=model_cfg['num_classes']
        )
    else: raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    avgloss_list = []
    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step, avgloss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            wandb,
            args
        )
        avgloss_list.append(avgloss)
        loss = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
                wandb,
                args
            )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    # plot loss
    tkey = time.strftime("%Y%m%d%H%M%S", time.localtime())
    plt.figure()
    plt.plot(range(len(avgloss_list)), avgloss_list, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('UNetPP')
    plt.legend()
    plt.savefig('./log/nn_isic_unetpp_e{}_t{}.png'.format(config.epochs, tkey))
    plt.close()

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
                wandb,
                args
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      
    if args.wandb:
        wandb.finish()
if __name__ == '__main__':
    config = setting_config
    main(config)