import numpy as np
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import csv
import os

import cv2
import scipy
from matplotlib import pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import spearmanr

def metric_cal(scores, gt_list, gt_mask_list, cal_pro=True):
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list, dtype=int)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    # img_roc_auc = roc_auc_score(gt_list, img_scores)
    img_roc_auc = 0
    # print('INFO: image ROCAUC: %.3f' % (img_roc_auc))

    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list, dtype=int)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten(),max_fpr=0.2)

    # calculate pro
    if cal_pro:
        pro_auc_score = cal_pro_metric(gt_mask_list, scores, fpr_thresh=0.2)
    else:
        pro_auc_score = 0

    return img_roc_auc, per_pixel_rocauc, pro_auc_score, threshold

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool_)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool_)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score


def learned_normalize(gt, v):
    max_idxs = []
    min_idxs = []
    max_vals = []
    min_vals = []
    max_vals_predicted = []
    min_vals_predicted = []
    gt_scores = [float(val[1]) for val in gt]
    predicted_scores = [float(val[1]) for val in v]
    max_val_gt = np.max(gt_scores)
    min_val_gt = np.min(gt_scores)
    N = 10
    for val in range(N):
        mx_idx = np.argmax(gt_scores)
        max_vals.append(gt_scores[mx_idx])
        max_vals_predicted.append(predicted_scores[mx_idx])
        max_idxs.append(mx_idx)
        gt_scores[mx_idx] = -1000000000

    gt_scores = [float(val[1]) for val in gt]
    for val in range(N):
        mn_idx = np.argmin(gt_scores)
        min_vals.append(gt_scores[mn_idx])
        min_vals_predicted.append(predicted_scores[mn_idx])

        min_idxs.append(mn_idx)
        gt_scores[mn_idx] = 1000000000

    max_mean_predicted = np.mean(max_vals_predicted)
    min_mean_predicted = np.mean(min_vals_predicted)
    max_mean = np.mean(max_vals)
    min_mean = np.mean(min_vals)

    # return ((predicted_scores - min_mean_predicted) / (max_mean_predicted - min_mean_predicted))*max_mean+min_mean
    return ((predicted_scores - min_mean_predicted) / (max_mean_predicted - min_mean_predicted+0.0000000001)) * (
            max_mean - min_mean) + min_mean


def calculate_pearson_correlation(v1, v2):
    mean1 = np.mean(v1)
    mean2 = np.mean(v2)
    std1 = np.std(v1)
    std2 = np.std(v2)
    cov_all = np.cov(v1, v2)
    # correlation = cov_all[0][0] / (std1 * std2)
    correlation_2 = pearsonr(np.transpose(v1), np.transpose(v2))
    # correlation_2 = spearmanr(np.transpose(v1), np.transpose(v2))
    return correlation_2

def calculate_spearman_correlation(v1, v2):
    mean1 = np.mean(v1)
    mean2 = np.mean(v2)
    std1 = np.std(v1)
    std2 = np.std(v2)
    cov_all = np.cov(v1, v2)
    # correlation = cov_all[0][0] / (std1 * std2)
    # correlation_2 = pearsonr(np.transpose(v1), np.transpose(v2))
    correlation_2 = spearmanr(np.transpose(v1), np.transpose(v2))
    return correlation_2

def plot_correlation_two_vectors(v1, v2,save=False):
    plt.clf()
    # result = calculate_pearson_correlation(v1[65:], v2[65:])
    # result = calculate_pearson_correlation(v1[0:64], v2[0:64])
    # result = calculate_pearson_correlation(v1[0:54], v2[0:54])
    result = calculate_pearson_correlation(v1, v2)  # pearson
    result_spear = calculate_spearman_correlation(v1, v2)  # spearman
    result_MSE = ((v1 - v2) ** 2).mean()  # MSE
    result_MAE = (np.abs(v1 - v2)).mean()  # MAE

    # hist=scipy.stats.histogram(v1, numbins=7, defaultlimits=(1,7), weights=None, printextras=False)

    # hist=scipy.stats.rv_histogram([v1], numbins=7, defaultlimits=None, weights=None, printextras=False)
    hist = np.histogram(v1, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    hist2 = np.histogram(v2, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    # hist = np.histogram(v1, bins=[1, 2, 3, 4, 5, 6, 7])
    # hist2 = np.histogram(v2, bins=[1, 2, 3, 4, 5, 6, 7])
    # hist2 = np.histogram(v2, bins=[1, 2, 3, 4, 5, 6, 7])
    try:
        result_W = scipy.stats.wasserstein_distance(np.arange(6), np.arange(6), hist[0], hist2[0])  # W-distance
    except:
        result_W=None
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(v1[0:int(len(v1) * 0.1)],
                                                                         v2[0:int(len(v1) * 0.1)])
    # np.random.shuffle(v1)
    # np.random.shuffle(v2)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(v1[0:int(len(v1))],
                                                                         v2[0:int(len(v1))])

    x1 = 0
    x2 = 8
    p1 = [x1, slope * x1 + intercept]
    p2 = [x2, slope * x2 + intercept]
    # result_2 = calculate_pearson_correlation(v1, v2)
    # plt.plot(scores, calculated_scores)
    cleft = True
    if (cleft):
        plt.scatter(v1[0:63], v2[0:63], alpha=0.5, label='normal facial images (generated)')
        plt.scatter(v1[64:], v2[64:], alpha=0.5, label='cleft samples (real)')
        # plt.plot([x1, x2], [p1[1], p2[1]])
    else:
        plt.scatter(v1[55:], v2[55:], alpha=0.5, label='normal facial images (generated)')
        plt.scatter(v1[0:54], v2[0:54], alpha=0.5, label='abnormal facial images (real)')

    # plt.plot(v1, v2, '-o')
    # plt.legend(bbox_to_anchor=(1, 1))
    plt.legend()
    # plt.title('human vs ' + loss + ' score correlation: ' + str(round(result[0], 2)), fontsize=16)
    plt.title('human vs AlexNet score correlation: ' + str(round(result[0], 2)), fontsize=16)
    plt.xlabel('Human Scores', fontsize=16)
    plt.ylabel('Calculated Scores', fontsize=16)
    plt.grid(True)
    if save:
        plt.savefig('res.png', dpi=100)
        return result[0],result_W,result_MAE,result_spear

    # plt.show()
    return result[0]


def write_res_csv(res, res_path,append=False):
    import csv
    if append:
        mode='a'
    else:
        mode='w'
    with open(res_path + '/results.csv', mode, newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # for i in res:
        #     spamwriter.writerow([str(i[0]), str(i[1])])
        spamwriter.writerow([str(res[0]), str(res[1])])
def calculate_best_correlation_offline(path):
    human_scores_path = 'IMAGES WITH 2O OR MORE RATINGS_MAY_27_2023_filtered.csv'
    human_scores_path = './IMAGES WITH 2O OR MORE RATINGS_MAY_27_2023_renamed.csv'
    human_scores_path = './ratings3.csv'
    # m_path = 'masks/mask.png'
    # m_path='masks/face_mask_eyes.png'
    m_path='masks/face_mask.png'
    mask = cv2.imread(m_path, 0)
    res=256
    mask = cv2.resize(mask, (res, res))

    dirs = sorted(os.listdir(path))
    correlations = []
    thresholds = []
    for t in np.linspace(0, 200, 200):
        scores = []
        # break

        for img in dirs:
            name = img.split('/')[-1]
            heatmap = cv2.imread(path + img, 0)
            heatmap = cv2.resize(heatmap, (res, res))
            heatmap[mask == 0] = 0
            heatmap[heatmap < t] = 0
            # heatmap[heatmap>=t]
            # scores.append([name,np.log(np.sum(heatmap)/(heatmap.shape[0]*heatmap.shape[1]/100))])
            # scores.append([name,np.log(np.sum(heatmap)/(heatmap.shape[0]*heatmap.shape[1]/100))])
            take_log=False
            if take_log:
                sum=np.sum(heatmap)
                if sum==0:
                    sum=0.0000001
                score=[name,np.log(sum/(heatmap.shape[0]*heatmap.shape[1]/100))]
            else:
                score=[name, np.sum(heatmap) / (heatmap.shape[0] * heatmap.shape[1] / 100)]
            scores.append(score)


            # write_res_csv(scores, '.')

        with open(human_scores_path, encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            # human_scores_names = sorted([[val] for val in list(csv_reader)])
            human_scores_names = sorted([[val[0], float(val[1])] for val in list(csv_reader)])

            human_scores = [float(val[1]) for val in human_scores_names]
        # human_scores_names.pop(8)
        human_scores = [float(val[1]) for val in human_scores_names]
        # scores = [float(val[1]) for val in scores]
        scores = learned_normalize(human_scores_names, scores)

        # plot_correlation_two_vectors(np.array(human_scores[77:]),np.array(scores[77:]))
        corr = -plot_correlation_two_vectors(np.array(human_scores), np.array(scores))

        if np.isnan(corr):
            break
        correlations.append(np.array(corr))
        thresholds.append(t)
        print(np.min(correlations), np.argmin(correlations), thresholds[np.argmin(correlations)])
        if len(correlations) > 2:
            if (corr < np.min(correlations[0:len(correlations) - 2])):
                pearson,wdis,mae,spear=plot_correlation_two_vectors(np.array(human_scores), np.array(scores), save=True)

        e = 0
    write_res_csv([np.min(correlations),thresholds[np.argmin(correlations)]], '.',append=True)
    # threshold and write the images into the heatmaps folder
    t= thresholds[np.argmin(correlations)]

    t = 33
    # t= 15 #83.2
    t= 18
    t= 25
    t= 38 #best bof nonclef whole mask 1024 train CDO hrnet48 alexnet l4 celebA-HQ
    t=30
    # t= 38
    # t= 45
    for img in dirs:
        name = img.split('/')[-1]
        heatmap = cv2.imread(path + img, 0)
        heatmap = cv2.resize(heatmap, (res, res))
        heatmap[mask == 0] = 0
        heatmap[heatmap < t] = 0
        heatmap[heatmap >= t] = 255

        # cv2.imwrite('heatmaps/' + name.split('/')[-1] + '.png', heatmap)
        cv2.imwrite('heatmaps/' + name.split('/')[-1], heatmap)

def calculate_best_correlation(heatmaps, names):
    human_scores_path = 'IMAGES WITH 2O OR MORE RATINGS_MAY_27_2023_filtered.csv'
    human_scores_path = './IMAGES WITH 2O OR MORE RATINGS_MAY_27_2023_renamed.csv'
    # human_scores_path = './ratings3.csv'
    m_path = 'masks/mask.png'
    # m_path='masks/face_mask_eyes.png'
    # m_path = 'masks/face_mask.png'
    mask = cv2.imread(m_path, 0)
    res = 256
    res = 512
    mask = cv2.resize(mask, (res, res))

    # dirs = sorted(os.listdir(path))
    correlations = []
    thresholds = []
    for t in np.linspace(0, 200, 200):
        scores = []
        # break

        for heatmap,name in zip(heatmaps, names):
            # name = img.split('/')[-1]
            # heatmap = cv2.imread(path + img, 0)
            heatmap = cv2.resize(heatmap, (res, res))
            heatmap[mask == 0] = 0
            heatmap[heatmap < t] = 0
            # heatmap[heatmap>=t]
            # scores.append([name,np.log(np.sum(heatmap)/(heatmap.shape[0]*heatmap.shape[1]/100))])
            # scores.append([name,np.log(np.sum(heatmap)/(heatmap.shape[0]*heatmap.shape[1]/100))])
            take_log = False
            if take_log:
                sum = np.sum(heatmap)
                if sum == 0:
                    sum = 0.0000001
                score = [name, np.log(sum / (heatmap.shape[0] * heatmap.shape[1] / 100))]
            else:
                score = [name, np.sum(heatmap) / (heatmap.shape[0] * heatmap.shape[1] / 100)]
            scores.append(score)

            # write_res_csv(scores, '.')

        with open(human_scores_path, encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            # human_scores_names = sorted([[val] for val in list(csv_reader)])
            human_scores_names = sorted([[val[0], float(val[1])] for val in list(csv_reader)])

            human_scores = [float(val[1]) for val in human_scores_names]
        # human_scores_names.pop(8)
        human_scores = [float(val[1]) for val in human_scores_names]
        # scores = [float(val[1]) for val in scores]
        scores = learned_normalize(human_scores_names, scores)

        # plot_correlation_two_vectors(np.array(human_scores[77:]),np.array(scores[77:]))
        corr = -plot_correlation_two_vectors(np.array(human_scores), np.array(scores))

        if np.isnan(corr):
            break
        correlations.append(np.array(corr))
        thresholds.append(t)
        print(np.min(correlations), np.argmin(correlations), thresholds[np.argmin(correlations)])
        if len(correlations) > 2:
            if (corr < np.min(correlations[0:len(correlations) - 2])):
                pearson, wdis, mae, spear = plot_correlation_two_vectors(np.array(human_scores),
                                                                         np.array(scores), save=True)

        e = 0
    write_res_csv([np.min(correlations), thresholds[np.argmin(correlations)]], '.', append=True)

    return np.min(correlations), thresholds[np.argmin(correlations)]
    # threshold and write the images into the heatmaps folder
    # t = thresholds[np.argmin(correlations)]

    # t = 33
    # # t= 15 #83.2
    # t = 18
    # t = 25
    # t = 38  # best bof nonclef whole mask 1024 train CDO hrnet48 alexnet l4 celebA-HQ
    # t = 30
    # # t= 38
    # # t= 45
    # for img in dirs:
    #     name = img.split('/')[-1]
    #     heatmap = cv2.imread(path + img, 0)
    #     heatmap = cv2.resize(heatmap, (res, res))
    #     heatmap[mask == 0] = 0
    #     heatmap[heatmap < t] = 0
    #     heatmap[heatmap >= t] = 255
    #
    #     # cv2.imwrite('heatmaps/' + name.split('/')[-1] + '.png', heatmap)
    #     cv2.imwrite('heatmaps/' + name.split('/')[-1], heatmap)

import torch


def find_params_size(model):
    param_size = 0
    n_params = 0
    import torchstat as stat
    # ex=torch.tensor((3,512,512)).to(device)
    # print(ex)
    # stat.stat(model, (3, 512, 512))
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        n_params += param.numel()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size = (buffer_size + param_size) / (1024 ** 2)
    return size, n_params


