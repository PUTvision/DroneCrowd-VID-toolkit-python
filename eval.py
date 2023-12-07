from src.save_anno_res import save_anno_res
from src.map import mAP


def main():
    threads = 10
    
    # Example usage:
    gt_path = "/home/host/Datasets/UP-COUNT/annotations/"
    det_path = "/home/host/Projects/UP-COUNT/UAV-DOT-DETECT/results/"

    list_path = "/home/host/Projects/UP-COUNT/UAV-DOT-DETECT/test_paths.txt"

    all_gt_result, all_det_result = save_anno_res(gt_path, det_path, list_path)

    # # calculate mAP

    mAP_results = mAP(all_gt_result, all_det_result, threads=threads)

    print(f'Average Precision@1:25\t = \t {mAP_results.get(range(1, 26)):.4f}')
    print(f'Average Precision@5\t = \t {mAP_results.get([5]):.4f}')
    print(f'Average Precision@10\t = \t {mAP_results.get([10]):.4f}')
    print(f'Average Precision@15\t = \t {mAP_results.get([15]):.4f}')
    print(f'Average Precision@20\t = \t {mAP_results.get([20]):.4f}')


if __name__ == '__main__':
    main()
