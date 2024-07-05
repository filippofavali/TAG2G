from Utils.gestureEvaluation import GestureEvaluator
from argparse import ArgumentParser
from VQVAE.vqvae_utils.motion_utils import load_bvh_file, create_TAG2G_pipeline
import joblib as jl
import os
import csv
import pandas as pd
import warnings
import numpy as np
import time
from tqdm import tqdm


warnings.filterwarnings('ignore')


class ResultsWriter:
    def __init__(self, save_dir, mode):

        self.save_dir = save_dir
        self.mode = mode
        self.csv_path = os.path.join(save_dir, f'000000_{mode}_scores.csv')
        self.stored_data = []

        # if mode == 'GHL':
        #     self.fieldnames = ['file_id', 'file_name', 'acc', 'jerk']
        #     with open(self.csv_path, mode='w') as file:
        #         writer = csv.DictWriter(file, fieldnames=self.fieldnames)
        #         writer.writeheader()
        # else:
        #     raise NotImplementedError(f'{mode} not implemented yet')

    def write_csv(self):

        self.pd_results = pd.DataFrame(self.stored_data)
        print(self.pd_results)
        self.pd_results.to_csv(self.csv_path, index=False, sep=';', decimal=',')
        print(f"Results correctly stored to: {self.csv_path}")

    def __call__(self, results_dict):

        # At call update current dicts
        self.stored_data.append(results_dict)

        # with open(self.csv_path, mode='a') as file:
        #     writer = csv.DictWriter(file, fieldnames=self.fieldnames)
        #     writer.writerow(results_dict)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--eval_path', type=str, default=None)
    parser.add_argument('--ma_path', type=str, default=None)
    parser.add_argument('--inter_path', type=str, default=None)
    parser.add_argument('--eval_mode', type=str, default=None, help='GHL, MAA, INA')
    args = parser.parse_args()

    pipeline = jl.load(r'utils\pipeline_expmap_74joints.sav')

    assert args.eval_path is not None, 'Provided an empty directory to be evaulated'
    assert args.eval_mode is not None, 'Provide eval as: GHL, MAA, INA'
    if args.eval_mode == 'INA':
        assert args.inter_path is not None, 'Provided an empty directory as interlocutor gesture'
    elif args.eval_mode == 'MAA':
        assert args.ma_path is not None, 'Provided an empy directory as main-agent gesture'

    # Compute tab 1 and 3 - a single model is evaluated
    evaluator = GestureEvaluator()
    results_writer = ResultsWriter(save_dir=args.eval_path, mode=args.eval_mode)

    pbar = tqdm(os.listdir(args.eval_path))
    for idx, filename in enumerate(pbar):
        pbar.set_description(f'Processing {filename}')
        # load gesture from the filename and compute the scores
        if '.bvh' in filename:

            try:

                # loading sample to be evaluated
                file = os.path.join(args.eval_path, filename)
                gesture = np.squeeze(pipeline.transform([load_bvh_file(file)]), axis=0).T

                if args.eval_mode == 'GHL':
                    result_dict = evaluator(gesture=gesture, mode=args.eval_mode)
                    result_dict['file_name'] = filename
                    results_writer(result_dict)

                if args.eval_mode == 'MAA':
                    # loading main-agent sample as a gt for evaluation
                    ma_file = os.path.join(args.ma_path, filename)
                    main_gesture = np.squeeze(pipeline.transform([load_bvh_file(ma_file)]), axis=0).T
                    result_dict = evaluator(gesture=gesture, gt_gesture=main_gesture, mode=args.eval_mode)
                    result_dict['file_name'] = filename
                    results_writer(result_dict)

                if args.eval_mode == 'INA':
                    #  loading interlocutor sample as a gt comparison
                    inter_file = os.path.join(args.inter_path, filename.replace('main-agent', 'interloctr'))
                    inter_gesture = np.squeeze(pipeline.transform([load_bvh_file(inter_file)]), axis=0).T
                    result_dict = evaluator(gesture=gesture, gt_gesture=inter_gesture, mode=args.eval_mode)
                    result_dict['file_name'] = filename
                    results_writer(result_dict)

            except FileNotFoundError:
                print(f'{filename} - file not found in interlocutor path')

    # At the end write results via the writer
    results_writer.write_csv()
