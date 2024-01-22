from typing import Tuple, List
import numpy as np
from multiprocessing import Process, Queue
import time
import queue  # imported for using queue.Empty exception
from scipy.spatial.distance import cdist


class mAP:
    def __init__(self, all_gt_result: dict, all_det_result: dict, thresh_range: Tuple[float, float, float], threads: int = 1) -> None:
        self.default_ranges = thresh_range
        self.results = {}

        print(f'Calculating mAP in range: {self.default_ranges.tolist()}')

        tasks_to_accomplish = Queue()
        tasks_that_are_done = Queue()
        processes = []

        for i in self.default_ranges:
            tasks_to_accomplish.put(i)

        # creating processes
        for w in range(threads):
            p = Process(target=self.do_job, args=(tasks_to_accomplish,
                        tasks_that_are_done, all_det_result, all_gt_result))
            processes.append(p)
            p.start()

        # completing process
        for p in processes:
            p.join()

        # print the output
        while not tasks_that_are_done.empty():
            result = tasks_that_are_done.get()
            self.results.update(result)

    def do_job(self, tasks_to_accomplish, tasks_that_are_done, all_det_result, all_gt_result):
        while True:
            try:
                '''
                    try to get task from the queue. get_nowait() function will 
                    raise queue.Empty exception if the queue is empty. 
                    queue(False) function would do the same task also.
                '''
                task = tasks_to_accomplish.get_nowait()

            except queue.Empty:

                break
            else:
                '''
                    if no exception has been raised, add the task completion 
                    message to task_that_are_done queue
                '''

                tasks_that_are_done.put(self._apply(
                    all_gt_result, all_det_result, task))
                time.sleep(.5)

        return True

    def _apply(self, all_gt_result: dict, all_det_result: dict, threshold: int):

        gtMatch = []
        detMatch = []

        for idSeq in sorted(list(all_gt_result.keys())):

            gt = all_gt_result[idSeq]
            det = all_det_result[idSeq]

            frs = np.int32(np.unique(gt[:, 0]))

            if det.shape[0] > 0:
                for f in frs:
                    idxGt = np.int32(gt[:, 0]) == f
                    idxDet = np.int32(det[:, 0]) == f

                    gt0 = gt[idxGt, 2:7]
                    dt0 = np.maximum(0, det[idxDet, 2:7])

                    gt1, dt1 = self._evalRes(gt0, dt0, threshold)

                    gtMatch.append(gt1[:, 4:5])
                    detMatch.append(dt1[:, 4:6])

        gtMatch = np.concatenate(gtMatch, axis=0)
        detMatch = np.concatenate(detMatch, axis=0)

        idrank = np.argsort(-detMatch[:, 0])

        tp = np.cumsum(np.int32(detMatch[idrank, 1]) == 1)
        fp = np.cumsum(np.int32(detMatch[idrank, 1]) == 0)

        prec = tp / np.maximum(1, (fp + tp))
        rec = tp / max(1, len(gtMatch))

        return {threshold: self._voc_ap(rec, prec) * 100}

    def get(self, ranges: List[int]):
        return np.mean([self.results[i] for i in ranges])

    def _evalRes(self, gt0, dt0, thr):
        # Check inputs
        if gt0.size == 0:
            gt0 = np.zeros((0, 5))
        if dt0.size == 0:
            dt0 = np.zeros((0, 5))

        assert dt0.shape[1] == 5
        nd = dt0.shape[0]
        assert gt0.shape[1] == 5
        ng = gt0.shape[0]

        # Sort dt highest score first, sort gt ignore last
        ord_dt = np.argsort(dt0[:, 4])[::-1]
        dt0 = dt0[ord_dt]

        ord_gt = np.argsort(gt0[:, 4])
        gt0 = gt0[ord_gt]

        gt = gt0
        dt = dt0

        dt = np.column_stack((dt, np.zeros(nd)))

        gt[:, 4] = -gt[:, 4]

        # Attempt to match each (sorted) dt to each (sorted) gt
        posdet = dt[:, :2] + dt[:, 2:4] / 2
        posgt = (gt[:, :2] + gt[:, 2:4]) / 2

        dis = cdist(posdet, posgt)

        for d in range(nd):
            bstOa = thr
            bstg = 0
            bstm = 0  # info about the best match so far

            for g in range(ng):
                # If this gt already matched, continue to the next gt
                m = gt[g, 4]
                if m == 1:
                    continue

                # If dt already matched, and on ignore gt, nothing more to do
                if bstm != 0 and m == -1:
                    break

                # Compute overlap area, continue to the next gt unless a better match is made
                if dis[d, g] > bstOa:
                    continue

                # Match successful and best so far, store appropriately
                bstOa = dis[d, g]
                bstg = g

                if m == 0:
                    bstm = 1
                else:
                    bstm = -1

            g = bstg
            m = bstm

            # Store the type of match for both dt and gt
            if m == -1:
                dt[d, 5] = m
            elif m == 1:
                gt[g, 4] = m
                dt[d, 5] = m

        return gt, dt

    def _voc_ap(self, rec, prec):
        mrec = np.concatenate(([0], rec, [1]))
        mpre = np.concatenate(([0], prec, [0]))

        for i in range(len(mpre) - 2, 0, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        return ap
