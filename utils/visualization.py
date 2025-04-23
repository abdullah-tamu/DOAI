import cv2

import numpy as np
import os

def normalize(v):
    return (v - np.min(v)) / (np.max(v) - np.min(v))
def plot_sample_cv2(names, imgs,final_heatmaps, scores_: dict, normalized_imgs, save_folder=None):
    # get subplot number
    total_number = len(imgs)

    scores = scores_.copy()
    # normalize anomalies
    for k, v in scores.items():
        max_value = np.max(v)
        min_value = np.min(v)

        normalizing =True
        if normalizing:
            scores[k] = (scores[k] - min_value) / max_value * 255
            scores[k] = scores[k].astype(np.uint8)
        else:
            scores[k] = np.array(scores[k])

    # save imgs
    for idx in range(total_number):
        for key in scores:
            heat_map = cv2.applyColorMap(np.array(scores[key][idx], dtype=np.uint8), cv2.COLORMAP_JET)
            visz_map = cv2.addWeighted(heat_map, 1, cv2.cvtColor(imgs[idx], cv2.COLOR_RGB2BGR), 0, 0)
            cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_{key}.jpg'),
                        visz_map)


    for idx,final_heatmap in enumerate(zip(final_heatmaps)):
        final_heatmap=normalize(final_heatmap)[0]*255
        heat_map = cv2.applyColorMap(np.array(final_heatmap, dtype=np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_final.jpg'),
                    heat_map)

        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_norm.jpg'),
                    cv2.cvtColor(normalized_imgs[idx], cv2.COLOR_RGB2BGR))

        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_org.jpg'),
                    cv2.cvtColor(imgs[idx], cv2.COLOR_RGB2BGR))




