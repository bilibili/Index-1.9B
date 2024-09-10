# flake8: noqa
# yapf: disable
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime
from itertools import product

import mmengine
from mmengine import ConfigDict
from tabulate import tabulate

from opencompass.partitioners.sub_naive import remove_duplicate_pairs
from opencompass.utils import dataset_abbr_from_cfg, model_abbr_from_cfg

from .utils import get_judgeanswer_and_reference, get_outdir


def model_abbr_from_cfg_used_in_summarizer(model):
    if model.get('summarizer_abbr', None):
        return model['summarizer_abbr']
    else:
        return model_abbr_from_cfg(model)

def post_process_compass_arena(s):
    if result := re.findall('(?:选择：|Choice: )([ABC])', s):
        return result[0]
    else:
        return None


def check_position_bias(judged_answers, references, banned_choice=['C']):
    """Check position bias for judgellm's judgement.

    Args:
        judged_answers: The successfully extracted judgement.
        references: The references contains original question, which is used to located the same question for different position judgement.
    """
    position_bias_flag = 0
    position_bias_dict = {}
    for judge, ref in zip(judged_answers, references):
        question = ref['question']
        question_hash = hash(question)
        if question_hash not in position_bias_dict:
            position_bias_dict[question_hash] = {
                'question': question,
                'judge': judge
            }
        else:
            first_judge = position_bias_dict[question_hash]['judge']
            if judge == first_judge and first_judge not in banned_choice and judge not in banned_choice:
                # If second choice is same with first choice, there has position bias.
                position_bias_flag += 1
    return position_bias_flag


class CompassArenaSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self,
                 config: ConfigDict,
                 judge_type='general',
                 check_pos_bias=True,
                 summary_type='single') -> None:
        self.tasks = []
        self.cfg = config
        self.base_models = self.cfg['datasets'][0]['base_models']
        self.compare_models = self.cfg['eval']['partitioner']['models']
        self.judge_models = self.cfg.get('judge_models', None)
        self.meta_judge_model = self.cfg.eval.partitioner.get('meta_judge_model', None)
        self.judge_type = judge_type
        assert self.judge_type in ['general']
        self.judge_map = {'general': post_process_compass_arena}
        self.judge_function = self.judge_map[self.judge_type]
        self.check_pos_bias = check_pos_bias
        self.summary_type = summary_type

    def get_score(self, time_str):
        output_dir, results_folder = get_outdir(self.cfg, time_str)
        model_combinations = list(product(self.base_models, self.compare_models))
        unique_combinations = remove_duplicate_pairs([combo for combo in model_combinations if combo[0] != combo[1]])

        if self.meta_judge_model is not None:
            self.judge_models.append(self.meta_judge_model)

        scores = {}

        for idx, judge_model_cfg in enumerate(self.judge_models):
            judge_model = model_abbr_from_cfg(judge_model_cfg)
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                for model_pair in unique_combinations:
                    model1 = model_pair[0]['abbr']
                    model2 = model_pair[1]['abbr']
                    if idx == len(self.judge_models):
                        subdir = model1 + '_' + model2 + '_summarized-by--' + judge_model
                    else:
                        subdir = model1 + '_' + model2 + '_judged-by--' + judge_model
                    subdir_path = os.path.join(results_folder, subdir)
                    if not os.path.isdir(subdir_path):
                        print(subdir_path + ' is not exist! please check!')
                        continue
                    judged_answers, references = get_judgeanswer_and_reference(dataset, subdir_path, self.judge_function)
                    if len(judged_answers) == 0:
                        scores[judge_model][dataset_abbr][model2] = {}
                        continue
                    if self.check_pos_bias:
                        bias_num = check_position_bias(judged_answers, references)
                    else:
                        bias_num = 0
                    win_model1 = defaultdict(float)
                    win_model2 = defaultdict(float)
                    categories = defaultdict(float)
                    model1 = references[0]['answer1']
                    model2 = references[0]['answer2']
                    for prediction, reference in zip(judged_answers, references):
                        categories[dataset_abbr] += 1
                        categories[reference['capability']] += 1

                        if prediction == 'A':
                            if reference['answer1'] == model1:
                                score_1, score_2 = 1, 0
                            else:
                                score_1, score_2 = 0, 1
                        elif prediction == 'B':
                            if reference['answer1'] == model1:
                                score_1, score_2 = 0, 1
                            else:
                                score_1, score_2 = 1, 0
                        elif prediction == 'C':
                            if self.summary_type == 'half_add':
                                score_1, score_2 = 0.5, 0.5
                            else:
                                score_1, score_2 = 0, 0

                        win_model1[reference['capability']] += score_1
                        win_model1[dataset_abbr] += score_1
                        win_model2[reference['capability']] += score_2
                        win_model2[dataset_abbr] += score_2
                    for capability in categories:
                        win_model1[capability] = win_model1[capability] / categories[capability] * 100
                        win_model1[capability] = round(win_model1[capability], 2)
                        win_model2[capability] = win_model2[capability] / categories[capability] * 100
                        win_model2[capability] = round(win_model2[capability], 2)

                    win_model1['position_bias'] = bias_num
                    win_model2['position_bias'] = bias_num

                    if judge_model not in scores:
                        scores[judge_model] = {}
                    if dataset_abbr not in scores[judge_model]:
                        scores[judge_model][dataset_abbr] = {}
                    scores[judge_model][dataset_abbr][model2] = win_model2

        return scores

    def summarize(
            self,
            time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S'),
    ):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """


        scores = self.get_score(time_str)
        # scores['win_' + model1] = win_model1
        output_dir, results_folder = get_outdir(self.cfg, time_str)

        all_scores = {}
        for idx, judge_model in enumerate(self.judge_models):
            score_by_judgemodel = {}
            judge_abbr = model_abbr_from_cfg(judge_model)
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                summarizer_model_abbrs = [model_abbr_from_cfg_used_in_summarizer(i) for i in self.compare_models]
                one_column = list(scores[judge_abbr][dataset_abbr].values())[0]
                row_headers = [i for i in one_column.keys() if i not in [dataset_abbr, 'position_bias']]
                row_headers = [dataset_abbr, 'position_bias'] + row_headers
                headers = [''] + summarizer_model_abbrs
                table = []
                for row_header in row_headers:
                    row = [row_header]
                    for model_cfg in self.compare_models:
                        model_abbr = model_abbr_from_cfg(model_cfg)
                        s = scores[judge_abbr][dataset_abbr][model_abbr].get(row_header, '')
                        if isinstance(s, float):
                            s = f'{s:.2f}'
                        if isinstance(s, int):
                            s = str(s)
                        row.append(s)
                    table.append(row)
                txt = tabulate(table, headers=headers)

                if idx == len(self.judge_models):
                    output_filename = osp.join(output_dir, dataset_abbr + '-summarized-by--' + judge_abbr + '-report.csv')
                else:
                    output_filename = osp.join(output_dir, dataset_abbr + '-judged-by--' + judge_abbr + '-report.csv')

                with open(output_filename, 'w') as f:
                    f.write(','.join(headers) + '\n')
                    for line in table:
                        f.write(','.join(line) + '\n')

            table = []
            summarizer_model_abbrs = [model_abbr_from_cfg_used_in_summarizer(i) for i in self.compare_models]
            headers = [''] + summarizer_model_abbrs
            for dataset in self.cfg['datasets']:
                dataset_abbr = dataset_abbr_from_cfg(dataset)
                row = [dataset_abbr]
                for model_cfg in self.compare_models:
                    model_abbr = model_abbr_from_cfg(model_cfg)
                    s = scores[judge_abbr][dataset_abbr][model_abbr].get(dataset_abbr, '')
                    if isinstance(s, float):
                        s = f'{s:.2f}'
                    if isinstance(s, int):
                        s = str(s)
                    row.append(s)
                table.append(row)
            txt = tabulate(table, headers=headers)

            if idx == len(self.judge_models):
                output_filename = osp.join(output_dir, 'compassarena-overall-summarized-by--' + judge_abbr + '.csv')
            else:
                output_filename = osp.join(output_dir, 'compassarena-overall-judged-by--' + judge_abbr + '.csv')

            table = [[row[0]] + [f'{x:.2f}' if not isinstance(x, str) else x for x in row[1:]] for row in table]
            with open(output_filename, 'w') as f:
                f.write(','.join(headers) + '\n')
                for line in table:
                    f.write(','.join(line) + '\n')

            for idx, model in enumerate(summarizer_model_abbrs):
                score_by_judgemodel[model] = {}
                for subset in table:
                    score_by_judgemodel[model][subset[0]] = subset[idx+1]
            all_scores[judge_abbr]=score_by_judgemodel
        return {'CompassArena': all_scores}
