import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial.distance import directed_hausdorff
import numpy as np

def test_img_classification(net_g, datatest, args, type = 'ce'):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    data_loader = DataLoader(datatest, batch_size=1)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
        # sum up batch loss
        if type == 'ce':
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        elif type == 'bce':
            # BCEWithLogitsLoss
            test_loss += F.binary_cross_entropy_with_logits(log_probs, target, reduction='sum').item()
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)

    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def tensor2np(tensor):
    tensor = tensor.squeeze().cpu()
    return tensor.detach().numpy()

def normtensor(tensor):
    tensor = torch.where(tensor<0., torch.zeros(1).cuda(), torch.ones(1).cuda())
    return tensor

def cal_iou(outputs, labels, SMOOTH=1e-6):
    with torch.no_grad():
        outputs = outputs.squeeze(1).bool()  # BATCH x 1 x H x W => BATCH x H x W
        labels = labels.squeeze(1).bool()
        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
        # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou

    # return iou.cpu().detach().numpy()

def get_iou_score(outputs, labels):
    A = labels.squeeze(1).bool()
    pred = torch.where(outputs<0., torch.zeros(1).cuda(), torch.ones(1).cuda())
    B = pred.squeeze(1).bool()
    intersection = (A & B).float().sum((1,2))
    union = (A| B).float().sum((1, 2)) 
    iou = (intersection + 1e-6) / (union + 1e-6)  
    return iou.cpu().detach().numpy()

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return torch.clamp(dice, 0.0, 1.0)  # 確保結果在 0 到 1 之間

def get_boundary(mask):
    # 确保mask是浮点类型
    mask = mask.float()
    
    # 使用膨胀和腐蚀操作来获取边界
    kernel = torch.ones(3, 3, device=mask.device)
    dilated = F.conv2d(mask, kernel.unsqueeze(0).unsqueeze(0), padding=1) > 0
    eroded = F.conv2d(mask, kernel.unsqueeze(0).unsqueeze(0), padding=1) == 9
    
    # 使用逻辑运算来获取边界
    boundary = (dilated & ~eroded).float()
    
    return boundary

def calculate_bme(pred_boundary, true_boundary):
    # 计算边界匹配误差
    diff = torch.abs(pred_boundary - true_boundary)
    bme = torch.sum(diff) / (torch.sum(true_boundary) + 1e-6)
    return bme

def calculate_mad(pred, mask):
    # 计算平均绝对距离
    diff = torch.abs(pred - mask)
    mad = torch.mean(diff)
    return mad

def calculate_hausdorff(pred, mask):
    # 计算Hausdorff距离
    pred_points = np.array(np.where(pred > 0.5)).T
    mask_points = np.array(np.where(mask > 0.5)).T
    
    if len(pred_points) == 0 or len(mask_points) == 0:
        return 0  # 如果其中一个为空，返回0
    
    forward_hausdorff = directed_hausdorff(pred_points, mask_points)[0]
    backward_hausdorff = directed_hausdorff(mask_points, pred_points)[0]
    
    return max(forward_hausdorff, backward_hausdorff)

def test_img_segmentation(model, device, testloader, loss_function):
    model.eval()
    running_loss = 0
    iou, dsc, precision, recall, f1_scores, bme_scores, mad_scores, hausdorff_distances = [], [], [], [], [], [], [], []
    testloader = DataLoader(testloader, batch_size=1)
    with torch.no_grad():
        for i, (input, mask) in enumerate(testloader):
            input, mask = input.to(device), mask.to(device)

            predict = model(input)
            loss = loss_function(predict, mask)
            running_loss += loss.item()
            
            # 现有的指标计算
            pred = normtensor(predict)
            iou.append(get_iou_score(predict, mask).mean())
            # 使用 f1_or_dsc 的計算方式替代 DSC
            true_positive = torch.sum((pred == 1) & (mask == 1)).float()
            false_positive = torch.sum((pred == 1) & (mask == 0)).float()
            false_negative = torch.sum((pred == 0) & (mask == 1)).float()
            f1_or_dsc = float(2 * true_positive) / float(2 * true_positive + false_positive + false_negative) if float(2 * true_positive + false_positive + false_negative) != 0 else 0
            f1_scores.append(f1_or_dsc)
            dsc.append(f1_or_dsc)
            # 如果没有这个小的常数，程序在某些情况下可能会崩溃或返回无穷大（inf），这会影响后续的计算和结果。因此，添加 1e-6 是一种常见的做法，用于提高数值稳定性。
            batch_precision = true_positive / (true_positive + false_positive + 1e-6)
            batch_recall = true_positive / (true_positive + false_negative + 1e-6)
 
            precision.append(batch_precision.item())
            recall.append(batch_recall.item())
            # 计算 BME
            pred_boundary = get_boundary(pred)
            mask_boundary = get_boundary(mask)
            bme = calculate_bme(pred_boundary, mask_boundary)
            bme_scores.append(bme.item())

            # 计算 MAD
            mad = calculate_mad(pred, mask)
            mad_scores.append(mad.item())

            # 计算 Hausdorff 距离
            hd = calculate_hausdorff(pred.cpu().numpy(), mask.cpu().numpy())
            hausdorff_distances.append(hd)

            # 记录第一张图像
            if ((i + 1) % 1) == 0:
                pred = normtensor(predict[0])
                img, pred, mak = tensor2np(input[0]), tensor2np(pred), tensor2np(mask[0])

    test_loss = running_loss / len(testloader)
    
    # Check for empty lists before calculating means
    mean_iou = np.mean(iou) if iou else 0
    mean_dsc = np.mean(dsc) if dsc else 0
    if mean_dsc < 0 or mean_dsc > 1:
        print(f"Warning: Invalid mean DSC value {mean_dsc}")    
    mean_precision = np.mean(precision) #if precision else 0
    mean_recall = np.mean(recall) #if recall else 0
    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    mean_bme = np.mean(bme_scores) #if bme_scores else 0
    mean_mad = np.mean(mad_scores) #if mad_scores else 0
    mean_hausdorff = np.mean(hausdorff_distances) if hausdorff_distances else 0
    # 如果当前的平均IoU（交并比）高于之前的最佳IoU
    # if mean_iou>best_iou:
    # # export to onnx + pt
    #     尝试导出模型    
    #     try:
    #         将模型导出为ONNX格式    
    #         torch.onnx.export(model, input, SAVE_PATH + RUN_NAME + '.onnx')
    #         保存模型的状态字典（权重）为PyTorch格式    
    #         torch.save(model.state_dict(), SAVE_PATH + RUN_NAME + '.pth')
    #     except:
    #         如果导出失败，打印错误信息
    #         print('Can export weights')
    return test_loss, mean_iou, mean_dsc, mean_precision, mean_recall, mean_f1, mean_bme, mean_mad, mean_hausdorff


def test_local_classification(net_g, data_loader, args, type='ce'):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # test_loss += F.cross_entropy(log_probs, target).item()
        if type == 'ce':
            test_loss += F.cross_entropy(log_probs, target).item()
        elif type == 'bce':
            # BCEWithLogitsLoss
            test_loss += F.binary_cross_entropy_with_logits(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    return correct, test_loss

def test_local_segmentation(model, device, testloader, loss_function):
    model.eval()
    running_loss = 0
    iou, dsc, precision, recall, f1_scores, bme_scores, mad_scores, hausdorff_distances = [], [], [], [], [], [], [], []
    testloader = DataLoader(testloader, batch_size=1)
    with torch.no_grad():
        for i, (input, mask) in enumerate(testloader):
            input, mask = input.to(device), mask.to(device)

            predict = model(input)
            loss = loss_function(predict, mask)
            running_loss += loss.item()
            
            # 现有的指标计算
            pred = normtensor(predict)
            iou.append(get_iou_score(predict, mask).mean())
            # 使用 f1_or_dsc 的計算方式替代 DSC
            true_positive = torch.sum((pred == 1) & (mask == 1)).float()
            false_positive = torch.sum((pred == 1) & (mask == 0)).float()
            false_negative = torch.sum((pred == 0) & (mask == 1)).float()
            f1_or_dsc = float(2 * true_positive) / float(2 * true_positive + false_positive + false_negative) if float(2 * true_positive + false_positive + false_negative) != 0 else 0
            f1_scores.append(f1_or_dsc)
            dsc.append(f1_or_dsc)
            # 如果没有这个小的常数，程序在某些情况下可能会崩溃或返回无穷大（inf），这会影响后续的计算和结果。因此，添加 1e-6 是一种常见的做法，用于提高数值稳定性。
            batch_precision = true_positive / (true_positive + false_positive + 1e-6)
            batch_recall = true_positive / (true_positive + false_negative + 1e-6)
 
            precision.append(batch_precision.item())
            recall.append(batch_recall.item())
            # 计算 BME
            pred_boundary = get_boundary(pred)
            mask_boundary = get_boundary(mask)
            bme = calculate_bme(pred_boundary, mask_boundary)
            bme_scores.append(bme.item())

            # 计算 MAD
            mad = calculate_mad(pred, mask)
            mad_scores.append(mad.item())

            # 计算 Hausdorff 距离
            hd = calculate_hausdorff(pred.cpu().numpy(), mask.cpu().numpy())
            hausdorff_distances.append(hd)

            # 记录第一张图像
            if ((i + 1) % 1) == 0:
                pred = normtensor(predict[0])
                img, pred, mak = tensor2np(input[0]), tensor2np(pred), tensor2np(mask[0])

    test_loss = running_loss / len(testloader)
    
    # Check for empty lists before calculating means
    mean_iou = np.mean(iou) #if iou else 0
    mean_dsc = np.mean(dsc) if dsc else 0
    if mean_dsc < 0 or mean_dsc > 1:
        print(f"Warning: Invalid mean DSC value {mean_dsc}")    
    mean_precision = np.mean(precision) #if precision else 0
    mean_recall = np.mean(recall) #if recall else 0
    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    mean_bme = np.mean(bme_scores) #if bme_scores else 0
    mean_mad = np.mean(mad_scores) #if mad_scores else 0
    mean_hausdorff = np.mean(hausdorff_distances) #if hausdorff_distances else 0
    # 如果当前的平均IoU（交并比）高于之前的最佳IoU
    # if mean_iou>best_iou:
    # # export to onnx + pt
    #     尝试导出模型    
    #     try:
    #         将模型导出为ONNX格式    
    #         torch.onnx.export(model, input, SAVE_PATH + RUN_NAME + '.onnx')
    #         保存模型的状态字典（权重）为PyTorch格式    
    #         torch.save(model.state_dict(), SAVE_PATH + RUN_NAME + '.pth')
    #     except:
    #         如果导出失败，打印错误信息
    #         print('Can export weights')
    return test_loss, mean_iou, mean_dsc, mean_precision, mean_recall, mean_f1, mean_bme, mean_mad, mean_hausdorff
