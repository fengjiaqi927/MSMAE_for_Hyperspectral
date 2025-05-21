import torch
from torch import nn
import train_utils.distributed_utils as utils

import time
from train_utils.distributed_utils import SmoothedValue

import matplotlib.pyplot as plt
import matplotlib
from skimage import color
import numpy as np
import random

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        target_modified = target.clone()
        target_modified[target_modified == 0] = 255
        losses[name] = nn.functional.cross_entropy(x, target_modified, ignore_index=255)
        # losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
        

    # 判断是否使用辅助分类器
    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target, _ in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # 只是用主分支的预测结果
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat

# 将十六进制颜色转换为BGR格式
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return (rgb[0], rgb[1], rgb[2])  # OpenCV uses BGR format


def plot(img_rgb, y_true, y_pred, path, fix_path):
    # colors = ('cyan', 'white', 'red', 'plum', 'darkviolet', 'magenta', 'yellow',
    #           'peru', 'darkkhaki', 'lime', 'yellowgreen', 'saddlebrown', 'darkslateblue')

    color_dic = {0: "#000000",
                 1: '#00FFFF',
                 2: '#FFFFFF',
                 3: '#FF0000',
                 4: '#DDA0DD',
                 5: '#9400D3',
                 6: '#FF00FF',
                 7: '#FFFF00',
                 8: '#CD853F',
                 9: '#BDB76B',
                 10: '#00FF00',
                 11: '#9ACD32',
                 12: '#8B4513',
                 13: '#483D8B'}

    label_dic = {0: 'Background',
                 1: 'Surface water',
                 2: 'Street',
                 3: 'Urban Fabric',
                 4: 'Industrial, commercial and transport',
                 5: 'Mine, dump and construction sites',
                 6: 'Artificial, vegetated areas',
                 7: 'Arable Land',
                 8: 'Permanent Crops',
                 9: 'Pastures',
                 10: 'Forests',
                 11: 'Shrub',
                 12: 'Open spaces with no vegetation',
                 13: 'Inland wetlands'}

    fig = plt.figure()
    # 图例
    # legend_handles = [
    #     matplotlib.lines.Line2D(
    #         [],
    #         [],
    #         marker="s",
    #         color="w",
    #         markerfacecolor=color_dic[yi],
    #         ms=10,
    #         alpha=1,
    #         linewidth=0,
    #         label=label_dic[yi],
    #         markeredgecolor="k",
    #     )
    #     for yi in label_dic.keys()
    # ]
    # legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    # plt.legend(handles=legend_handles, **legend_kwargs_)

      
    plt.subplot(1, 3, 1)
    plt.title('img_rgb')
    plt.imshow(img_rgb)

    # image 1
    plt.subplot(1, 3, 2)
   
    
    height, width = y_true.shape
    dst_train = np.zeros((height, width, 3), dtype=np.uint8)
    for _label, _color in color_dic.items():
        bgr_color = hex_to_bgr(_color)
        dst_train[y_true == _label] = bgr_color  
    
    # dst_train = color.label2rgb(np.array(y_true), colors=colors, bg_label=0)
    
    plt.title('ground truth')
    plt.imshow(dst_train)
    # from skimage import io 
    # io.imsave('multi_train/C2Seg/test_vis/label_map_recover_test.jpg',  np.uint8(dst_train))
    # io.imsave(os.path.join('./data', 'train_label_map.jpg'), dst_train)
    # io.show()

    plt.subplot(1, 3, 3)

    dst_test = np.zeros((height, width, 3), dtype=np.uint8)
    for _label, _color in color_dic.items():
        bgr_color = hex_to_bgr(_color)
        dst_test[y_pred == _label] = bgr_color  
    
    # dst_test = color.label2rgb(np.array(y_pred), colors=colors, bg_label=0)
    plt.title('predict')
    plt.imshow(dst_test)
    # io.imsave(os.path.join('./data', 'test_label_map.jpg'), dst_test)
    # io.show()
    # fig.savefig(os.path.join(path, "label_map_recover.jpg"), dpi=600, bbox_inches='tight')
    fig.savefig(path, dpi=600, bbox_inches='tight')
    fig.savefig(fix_path, dpi=600, bbox_inches='tight')
    
    plt.close()

def evaluate_without_bg(model, data_loader, device, num_classes, vis_path, epoch):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes-1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        flag=0
        random_num = random.randint(1, 17)
        for data_iter_step, (image, target, id) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        # for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # 只是用主分支的预测结果
            output = output['out']
            pred = output.argmax(dim=1)
            
            # 1. 构造掩码：保留标签 ≠ 0 的位置
            valid_mask = (target != 0)
            # 2. 使用掩码过滤数据
            target_filtered = target[valid_mask].clone()
            pred_filtered = pred[valid_mask].clone()
            # 3. 让原来的标签 [1..(num_classes-1)] → [0..(num_classes-2)]
            #   类似地预测值也做同样的偏移
            target_filtered = target_filtered - 1
            pred_filtered   = pred_filtered - 1

            # 4. 过滤负数
            positive_mask = (pred_filtered >= 0)
            target_filtered = target_filtered[positive_mask]
            pred_filtered = pred_filtered[positive_mask]
            # 5. 更新混淆矩阵
            try:
                confmat.update(target_filtered.long().flatten(), pred_filtered.long().flatten())
            except:
                print(max(target_filtered.long().flatten()),min(target_filtered.long().flatten()))
                print(max(pred_filtered.long().flatten()),min(pred_filtered.long().flatten()))

            if flag==0 and (data_iter_step+1) % random_num == 0:
                if random_num %2 == 0:
                    from sklearn.decomposition import PCA
                    hsi_matrix = np.reshape(np.transpose(image[0, :, :, :].cpu().numpy(),(1, 2, 0)), (image.shape[2] * image.shape[3], image.shape[1]))  # 2456*811 242
                    pca = PCA(n_components=3)
                    pca.fit_transform(hsi_matrix)
                    newspace = pca.components_
                    newspace = newspace.transpose()  # 242*72
                    hsi_matrix = np.matmul(hsi_matrix, newspace) 
                    img_rgb = np.reshape(hsi_matrix, (image.shape[2] , image.shape[3], pca.n_components_))
                    img_rgb = np.clip(img_rgb*255, 0, 255).astype(np.uint8)
                else:
                    import cv2
                    msi = 'berlin/' if 'test' in vis_path else 'augsburg/'
                    msi = '/dev1/fengjq/dataset/crosscity_data/vis/' + msi + 'msi/train_img_' + id[0] + '.png'
                    img_rgb = cv2.imread(msi)
                img_rgb = np.clip(np.sqrt(np.transpose(image[0, [7,13,19], :, :].cpu().numpy(),(1, 2, 0))) * 255, 0, 255).astype(np.uint8)
                # img_rgb =  (np.transpose(image[0, [7,13,19], :, :].cpu().numpy(),(1, 2, 0))* 255).astype(np.uint8)
                img_rgb = np.sqrt(img_rgb / 255.0) * 255.0
                img_rgb = img_rgb.astype(np.uint8)
                plot(img_rgb, target.cpu()[0,:,:], pred.cpu()[0,:,:], vis_path  + '{}_'.format(epoch) + id[0] + '_label_map_recover.jpg', vis_path + 'label_map_recover.jpg')
                flag = 1
            
            # img_rgb =  (np.transpose(image[0, [7,13,19], :, :].cpu().numpy(),(1, 2, 0))* 255).astype(np.uint8)
            # img_rgb = np.sqrt(img_rgb / 255.0) * 255.0
            # img_rgb = img_rgb.astype(np.uint8)
            
            # plot(img_rgb, target.cpu()[0,:,:], pred.cpu()[0,:,:], vis_path  + '{}_'.format(epoch) + id[0] + '_label_map_recover.jpg', vis_path + 'label_map_recover.jpg')
            
            # color_dic = {
            #     0: "#000000", 1: '#00FFFF', 2: '#FFFFFF', 3: '#FF0000', 4: '#DDA0DD',
            #     5: '#9400D3', 6: '#FF00FF', 7: '#FFFF00', 8: '#CD853F', 9: '#BDB76B',
            #     10: '#00FF00', 11: '#9ACD32', 12: '#8B4513', 13: '#483D8B'
            # }
            # # 将十六进制颜色转换为BGR格式
            # def hex_to_bgr(hex_color):
            #     hex_color = hex_color.lstrip('#')
            #     rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
            #     return (rgb[2], rgb[1], rgb[0])  # OpenCV uses BGR format
            # # 创建一个空的 RGB 图像（高度 x 宽度 x 3）
            # height, width = target[0].shape
            # subset_label = target.cpu()[0,:,:]
            # label_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            # # 根据 color_dic 填充图像
            # for _label, _color in color_dic.items():
            #     bgr_color = hex_to_bgr(_color)
            #     label_rgb[subset_label == _label] = bgr_color     
            # import cv2 
            # cv2.imwrite(vis_path  + '{}_'.format(epoch) + id[0] + '_label.jpg', label_rgb)
            # '/dev1/fengjq/dataset/crosscity_data/vis/'
            # msi = 'berlin/' if 'test' in vis_path else 'augsburg/'
            # msi = '/dev1/fengjq/dataset/crosscity_data/vis/' + msi + 'msi/train_img_' + id[0] + '.png'
            # msi_img = cv2.imread(msi)
            
            # print("id")
    

        confmat.reduce_from_all_processes()

    return confmat    

accumiter=16
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, total_length, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    i = 0
    for image, target, id in metric_logger.log_every(data_loader, print_freq, header):
        try:
            image, target = image.to(device), target.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                # print(torch.max(target),torch.min(target))
                loss = criterion(output, target)

            # loss = loss / accumiter
            # loss.backward()
            # if(i%accumiter)==0:
            #    lr_scheduler.step()
            # if((i+1)%accumiter)==0:
            #     optimizer.step()
            #     optimizer.zero_grad()



            # 梯度清零
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:

                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)
            i+=1
            # print(i)
        except Exception as e:
            print("Error:", e)



    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=10,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    # torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)
    # optimizer：被调整学习率的优化器
    # lr_lambda：用户自定义的学习率调整规则。可以是lambda表达式，也可以是函数
    # last_epoch：当前优化器的已迭代次数，后文我们将其称为epoch计数器。默认是-1，字面意思是第-1个epoch已完成，也就是当前epoch从0算起，从头开始训练。如果是加载checkpoint继续训练，那么这里要传入对应的已迭代次数
    # verbose：是否在更新学习率时在控制台输出提醒
