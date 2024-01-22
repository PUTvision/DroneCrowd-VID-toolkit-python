import click
import os
from pathlib import Path
import numpy as np

from src.save_anno_res import save_anno_res
from src.map import mAP


@click.command()
@click.option('--dataset', '-d', type=click.Choice(['dronecrowd', 'upcount'], case_sensitive=False), help='Dataset name', required=True)
@click.option('--preds', '-p', type=click.Path(exists=True), help='Path to predictions', required=True)
@click.option('--threads', '-t', type=int, default=10, help='Number of threads')
def main(dataset, preds, threads):
        
    if dataset == 'dronecrowd':    
        gt_path = os.path.join(Path(__file__).parent, 'gt', 'dronecrowd')
        list_path = os.path.join(Path(__file__).parent, 'gt', 'dronecrowd_testlist.txt')
        img_wh = (1920, 1080)
    elif dataset == 'upcount':
        gt_path = os.path.join(Path(preds).parent, 'gt', 'upcount')
        list_path = os.path.join(Path(preds).parent, 'gt', 'upcount_testlist.txt')
        img_wh = (3840, 2160)

    all_gt_result, all_det_result = save_anno_res(gt_path, preds, list_path, dataset=dataset)

    # calculate mAP

    thresh_range = np.arange(1, 26, 1)
    mAP_results = mAP(all_gt_result, all_det_result, thresh_range=thresh_range, threads=threads)

    print(f'Average Precision@1:25\t = \t {mAP_results.get(thresh_range):.4f}')
    print(f'Average Precision@5\t = \t {mAP_results.get([5]):.4f}')
    print(f'Average Precision@10\t = \t {mAP_results.get([10]):.4f}')
    print(f'Average Precision@15\t = \t {mAP_results.get([15]):.4f}')
    print(f'Average Precision@20\t = \t {mAP_results.get([20]):.4f}')
    
    thresh_range_percent = np.arange(min(img_wh) * 0.005, max(img_wh) * 0.031, min(img_wh) * 0.005)
    mAP_results_percent = mAP(all_gt_result, all_det_result, thresh_range=thresh_range_percent, threads=threads)
    
    print(f'Average Precision@0.5-3.0%\t = \t {mAP_results_percent.get(thresh_range_percent):.4f}')
    print(f'Average Precision@1.0%\t = \t {mAP_results_percent.get([min(img_wh) * 0.01]):.4f}')
    print(f'Average Precision@2.0%\t = \t {mAP_results_percent.get([min(img_wh) * 0.02]):.4f}')
    print(f'Average Precision@3.0%\t = \t {mAP_results_percent.get([min(img_wh) * 0.03]):.4f}')
    

if __name__ == '__main__':
    main()
