#%%
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from IPython.core.display import display, HTML
from urllib.parse import quote
import cv2
from pathlib import Path
import pandas as pd
import os
import numpy as np
import json

#%%
folder = "unet_bce_4"

ROOT = "/thunderdisk"
mask_folder = "/thunderdisk/data_rene_policistico_2/all_masks"  # ROOT / 'data_rene_policistico_2'/'all_masks'
res_model_folder = "/thunderdisk/data_rene_policistico_2/predictions/all_images"  # ROOT / 'data_rene_policistico_2'/ 'predictions' /'all_images'

avg = "micro"

#%%

def missed_wrong_cysts(gt, pred):
    detected = 0
    wrong = 0
    missed = 0
    gt_contours, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    total = len(gt_contours)

    for c in gt_contours:
        single_gt = np.zeros_like(gt)
        cv2.fillPoly(single_gt, pts=[c], color=(1))

        if np.logical_and(single_gt, pred).any():
            detected += 1
        else:
            missed += 1

    for c in pred_contours:
        single_pred = np.zeros_like(pred)
        cv2.fillPoly(single_pred, pts=[c], color=(1))

        if not np.logical_and(single_pred, gt).any():
            wrong += 1

    return total, detected, missed, wrong


def open_mask(path):
    im = str(path)
    # print(type(cv2.imread(im, cv2.IMREAD_GRAYSCALE))
    return cv2.imread(im, cv2.IMREAD_GRAYSCALE)


def clean_mask(img, thr=58):
    blank = np.zeros_like(img)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > thr:
            cv2.fillPoly(blank, pts=[c], color=(255))

    return blank


def write_results(n_tiles=4, custom_name=None):
    print(f"Writing for {n_tiles}x{n_tiles}")
    folder = f"report_{n_tiles}"
    if custom_name:
        folder = custom_name
    datafile = f"results/{folder}.csv"
    C = []
    for pred in os.listdir(res_model_folder):
        # if pred.is_dir(): continue
        name = os.path.splitext(pred)[0]
        print('name',name)
        s = pd.Series([], dtype=pd.StringDtype())
        s.name = name

        gt = open_mask(f"{mask_folder}/{name}.png")

        pred_img = open_mask(os.path.join(res_model_folder,pred))
        pred_img = clean_mask(pred_img)

        #     plt.imshow(pred_img)

        s["# cysts"], s["# detected"], s["# missed"], s["# wrong"] = missed_wrong_cysts(
            gt, pred_img
        )
        s["# recall"] = s["# detected"] / (s["# detected"] + s["# missed"] + 0.0001)
        s["iou"] = jaccard_score(gt, pred_img, average=avg)
        s["precision"] = precision_score(gt, pred_img, average=avg, zero_division=1)
        s["recall"] = recall_score(gt, pred_img, average=avg, zero_division=1)
        #     s['f1_score'] = f1_score(gt, pred_img, average=avg, zero_division=1)
        # fp fn recall
        C.append(s)

    print(C)

    df_bench = pd.DataFrame(C).to_csv(datafile)
    # json.dump(df_bench, open(datafile, "w"))
    return

#%%
write_results(custom_name=folder)

# %%
