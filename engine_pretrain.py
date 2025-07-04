# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import wandb

import util.misc as misc
import util.lr_sched as lr_sched

import cv2
import numpy as np


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, (samples, mask_UM) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        if args.mask_UM_flag is not None:
            mask_UM = mask_UM.to(device, non_blocking=True).flatten(1).to(torch.bool)
        else:
            mask_UM = None
        with torch.cuda.amp.autocast():
            if args.mask_UM_flag is not None:
                temp_mask_ratio = round(float(1-(1-args.mask_ratio)/(1-0.75)),2) # (1-0.75) UM * (1-0.6) RM = (1-0.9) mask
            loss, imgs, pred, loss_list = model(samples, mask_ratio=temp_mask_ratio, mask_UM=mask_UM)
            if data_iter_step % print_freq == 0:

                img_rgb =  (np.transpose(imgs[0, [7,13,19], :, :].cpu().numpy(),(1, 2, 0))* 255).astype(np.uint8)
                pred_rgb = (np.transpose(pred.squeeze(1)[0, [7,13,19], :, :].cpu().detach().numpy(),(1, 2, 0))* 255).astype(np.uint8)
                
                img_rgb = np.sqrt(img_rgb / 255.0) * 255.0
                pred_rgb = np.sqrt(pred_rgb / 255.0) * 255.0
                # cv2.imwrite(args.log_dir + '/vis/'+ '{}_{}_output_image_1.png'.format(epoch,data_iter_step), img[:,:,1].astype(np.uint8))
                # cv2.imwrite(args.log_dir + '/vis/'+ '{}_{}_output_pred_1.png'.format(epoch,data_iter_step), pred[:,:,1].astype(np.uint8))
                cv2.imwrite(args.log_dir + '/vis/'+ '{}_{}_output_image.png'.format(epoch,data_iter_step), img_rgb.astype(np.uint8))
                cv2.imwrite(args.log_dir + '/vis/'+ '{}_{}_output_pred.png'.format(epoch,data_iter_step), pred_rgb.astype(np.uint8))
                cv2.imwrite(args.log_dir + '/vis/'+ 'output_image.png'.format(epoch,data_iter_step), img_rgb.astype(np.uint8))
                cv2.imwrite(args.log_dir + '/vis/'+ 'output_pred.png'.format(epoch,data_iter_step), pred_rgb.astype(np.uint8))
                img_mean = imgs[0,:,:,:].reshape(-1, img_rgb.shape[0] * img_rgb.shape[1]).mean(dim=-1).cpu().numpy()
                img_sample = imgs[0,:,:,:].reshape(-1, img_rgb.shape[0] * img_rgb.shape[1])[:,int(img_rgb.shape[0] * img_rgb.shape[1]/2)].cpu().numpy()
                pred_mean = pred.squeeze(0)[0,:,:,:].reshape(-1, img_rgb.shape[0] * img_rgb.shape[1]).mean(dim=-1).cpu().detach().numpy()
                pred_sample = pred.squeeze(0)[0,:,:,:].reshape(-1, img_rgb.shape[0] * img_rgb.shape[1])[:,int(img_rgb.shape[0] * img_rgb.shape[1]/2)].cpu().detach().numpy()
                import matplotlib.pyplot as plt
                plt.switch_backend('agg')
                plt.figure(figsize=(10, 6))
                plt.plot(img_mean, label='img_mean', linewidth=2, linestyle='-', color='C0')
                plt.plot(img_sample, label='img_sample', linewidth=2, linestyle='-', color='C1')
                plt.plot(pred_mean, label='pred_mean', linewidth=2, linestyle='--', color='C0')
                plt.plot(pred_sample, label='pred_sample', linewidth=2, linestyle='--', color='C1')
                plt.title('Comparison of Four Similar Curves')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                save_path = args.log_dir + '/vis/'+ '{}_{}_output_plt.png'.format(epoch,data_iter_step)
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.savefig(args.log_dir + '/vis/'+ 'output_plt.png'.format(epoch,data_iter_step), dpi=100, bbox_inches='tight')
                plt.close()






            
        loss_value = loss.item()
        loss_list_item = [round(i.item(), 2) for i in loss_list]

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")
            # sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_1=loss_list_item[0])
        metric_logger.update(loss_2=loss_list_item[1])
        metric_logger.update(loss_3=loss_list_item[2])

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            # Wandb logging
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_temporal(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, timestamps, _) in \
            enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, timestamps, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            # Use wandb
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}