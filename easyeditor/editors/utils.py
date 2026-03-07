from typing import Optional, Union, List, Tuple, Dict
import os
import json
import numpy as np
import random
import math

def _chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]
        
def get_all_acc_keys(dict_list):
    all_keys = set()

    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)
                
    for dictionary in dict_list:
        recursive_keys(dictionary)

    return all_keys

def _flatten_numeric_values(value):
    numeric_values = []

    if value is None:
        return numeric_values

    if isinstance(value, dict):
        return numeric_values

    if isinstance(value, (list, tuple)):
        for item in value:
            numeric_values.extend(_flatten_numeric_values(item))
        return numeric_values

    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return numeric_values

    if np.isnan(scalar):
        return numeric_values

    numeric_values.append(scalar)
    return numeric_values

def _mean_numeric_values(values):
    numeric_values = []
    for value in values:
        numeric_values.extend(_flatten_numeric_values(value))

    if len(numeric_values) == 0:
        return None

    return float(np.mean(numeric_values))

def _collect_nested_acc_metrics(all_metrics, eval_split, section_key):
    nested_summary = dict()
    acc_keys = set()

    for metric in all_metrics:
        section = metric.get(eval_split, {}).get(section_key, {})
        if isinstance(section, dict):
            for key in section.keys():
                if key.endswith('acc'):
                    acc_keys.add(key)

    for acc_key in sorted(acc_keys):
        mean_value = _mean_numeric_values([
            metric.get(eval_split, {}).get(section_key, {}).get(acc_key)
            for metric in all_metrics
            if isinstance(metric.get(eval_split, {}).get(section_key, {}), dict)
            and acc_key in metric.get(eval_split, {}).get(section_key, {})
        ])
        if mean_value is not None:
            nested_summary[acc_key] = mean_value

    return nested_summary
    
def summary_metrics(all_metrics):
    if isinstance(all_metrics, dict):
        all_metrics = [all_metrics, ]
    all_metrics = [metric for metric in all_metrics if isinstance(metric, dict)]

    if len(all_metrics) == 0:
        print("Metrics Summary: {}")
        return {}

    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    output_file = os.path.join(logs_dir, 'results.json')
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)

    mean_metrics = dict()
    scalar_metric_keys = ["rewrite_acc", "rephrase_acc", 'rewrite_ppl', 'ood_acc']

    for eval in ["pre", "post"]:
        eval_metrics = [metric[eval] for metric in all_metrics if isinstance(metric.get(eval), dict)]
        if len(eval_metrics) == 0:
            continue

        mean_metrics[eval] = dict()
        for key in scalar_metric_keys:
            mean_value = _mean_numeric_values([
                metric[eval][key]
                for metric in all_metrics
                if isinstance(metric.get(eval), dict) and key in metric[eval]
            ])
            if mean_value is not None:
                mean_metrics[eval][key] = mean_value

        for key in ["locality", "portability"]:
            nested_metrics = _collect_nested_acc_metrics(all_metrics, eval, key)
            if len(nested_metrics) > 0:
                mean_metrics[eval][key] = nested_metrics
    # mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])

    print("Metrics Summary: ", mean_metrics)
    return mean_metrics

def _prepare_requests(prompts: Union[str, List[str]],
                      target_new: Union[str, List[str]],
                      ground_truth: Union[str, List[str]],
                      target_neg: Optional[Union[str, List[str]]] = None,
                      rephrase_prompts: Optional[Union[str, List[str]]] = None,
                      locality_inputs: Optional[Dict] = None,
                      portability_inputs: Optional[Dict] = None,
                      **kwargs
                      ):

    requests = [{
        'prompt': prompt,
        'target_new': target_new_,
        'ground_truth': ground_truth_,
        'portability': {},
        'locality': {}
    }
    for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
    ]

    if target_neg is not None:
        if isinstance(target_neg, str):
            target_neg = [target_neg,]
        assert len(target_neg) == len(prompts)
        for i, request in enumerate(requests):
            request.update(
                {
                    'target_neg': target_neg[i]
                }
            )

    if 'subject' in kwargs:
        if isinstance(kwargs['subject'], str):
            kwargs['subject'] = [kwargs['subject'],]
        else:
            assert len(kwargs['subject']) == len(prompts)
        for prompt_, subject_ in zip(prompts, kwargs['subject']):
            assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

        for i, request in enumerate(requests):
            request.update(
                {
                    'subject': kwargs['subject'][i]
                }
            )
    if 'loc_prompts' in kwargs:
        if isinstance(kwargs['loc_prompts'], str):
            kwargs['loc_prompts'] = [kwargs['loc_prompts'],]
        if len(kwargs['loc_prompts']) < len(requests):
            kwargs['loc_prompts'] = (kwargs['loc_prompts'] * math.ceil(len(requests) / len(kwargs['loc_prompts'])))[:len(requests)]
            random.shuffle(kwargs['loc_prompts'])
        assert len(kwargs['loc_prompts']) == len(prompts)

        for i, request in enumerate(requests):
            request.update(
                {
                    'loc_prompt': kwargs['loc_prompts'][i]
                }
            )

    if rephrase_prompts is not None:
        if isinstance(rephrase_prompts, str):
            rephrase_prompts = [rephrase_prompts,]

        for i, request in enumerate(requests):
            request.update(
                {
                    'rephrase_prompt': rephrase_prompts[i],
                }
            )
    if locality_inputs is not None:
        for locality_key in locality_inputs.keys():
            if isinstance(locality_inputs[locality_key]['prompt'], str):
                locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
            assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
            == len(requests), print('One Edit instance needs one locality input.....')

            for i, request in enumerate(requests):
                if locality_inputs[locality_key]['prompt'][i] is not None:
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )

    if portability_inputs is not None:
        for portability_key in portability_inputs.keys():
            if isinstance(portability_inputs[portability_key]['prompt'], str):
                portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
            assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
            == len(requests), 'One Edit instance needs one portability input.....'

            for i, request in enumerate(requests):
                if portability_inputs[portability_key]['prompt'][i] is not None:
                    request['portability'].update(
                        {
                            portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }
                        }
                    )
    return requests
