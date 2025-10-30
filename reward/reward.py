import json
import math_utils
import nest_asyncio
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import asyncio
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

if __name__ == "__main__":

    config = get_config()

    project_name = config.experiment.project
    
    
    

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model.model_path

    outputs_name = "eval-" + pretrained_model.split("/")[-3] + "-" + dataset
    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    index_list = []
    extracted_output_list = []
    ground_truth_list = []
    response_length_list = []
    for i in range(len(data)):
        
        response_length_list = response_length_list + data[i]["response_length"]
        index_list = index_list + [i] * len(data[i]["extracted_output"])
        extracted_output_list = extracted_output_list + data[i]["extracted_output"]
        if config.dataset.data_type == "math":
            data[i]["correctness"] = []
            ground_truth_list = ground_truth_list + [data[i]["ground_truth_answer"]] * len(data[i]["extracted_output"])

    

    if config.dataset.data_type == "math":

        nest_asyncio.apply()

        async def get_correctness():
            executor = ThreadPoolExecutor(max_workers=64)
            tasks = []
            for i in range(len(index_list)):
                tasks.append(math_utils.is_equal(extracted_output_list[i], ground_truth_list[i], executor))
            results = await asyncio.gather(*tasks)
            return results
    
        correctness_list = asyncio.run(get_correctness())
        for i in range(len(index_list)):
            index_i = index_list[i]
            data[index_i]["correctness"].append(correctness_list[i])



    def z_score_normalize(lst):
        mean = sum(lst) / len(lst)
        std = (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5
        if std == 0:
            return [0 for x in lst]
        return [(x - mean) / std for x in lst]






    def set_last_t(lst: list, t: int) -> None:
        new_lst = lst.copy()
        new_val = max(lst) + 1
        new_lst[-t:] = [new_val] * t
        return new_lst


    if config.dataset.data_type == "math":
        # acc = sum(correctness_list)/len(correctness_list)
        num_k = len(data[0]["extracted_output"])
        best_of_k_acc = []

        for k in range(1, num_k + 1):
            hits = []
            for sample in data:
                if len(sample["correctness"]) == 0:
                    continue
                # 只看前 k 个回答
                topk_correct = sample["correctness"][:k]
                # 有一个正确就算命中
                hits.append(any(topk_correct))
            acc_k = sum(hits) / len(hits)
            best_of_k_acc.append(acc_k)

    else:
        num_task   = 0
        num_correct_task = 0
        for x in data:
            for y in x["correctness"]:
                num_correct_task += all(y)
                num_task += 1
        acc = num_correct_task / num_task if num_task else 0

    if config.rollout.output_unmasking_history == False:
        for i in range(len(data)):
            data[i]["step_map"] = []
    
    import os
    
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        # Save + print
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        
        
        avg_len = sum(response_length_list)/len(response_length_list)
        print("================ Evaluation Results ==================")
        save_and_print(f"avg length: {avg_len}")
        for k, acc_k in enumerate(best_of_k_acc, start=1):
            save_and_print(f"Best-of-{k}: {acc_k:.4f}")
