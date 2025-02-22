import argparse
import subprocess
import json

# Step 1: Define parameter groups with iterable arrays
parameter_groups = {
    
    # sc dataset
    
    "default": {
        ###
        "model_id":"mitecg",
        "data": "mitecg",
        "enc_in": 1,
        "num_class": 2,
        "seq_len":360,
        "num_prototypes":[4],
        "prototype_len":30,
        "two_class": 0,
        "is_training": 0,
        "patience": 1,
        "train_epochs": 1,
        "batch_size": 16,
        "prototype_init": "randn" # "kaiming"   
    },
    
        "ab": {
        ###
        "model_id":"mitecg",
        "data": "mitecg",
        "enc_in": 1,
        "num_class": 2,
        "seq_len":360,
        "num_prototypes":4,
        "prototype_len":30,
        "two_class": 0,
        "is_training": 0,
        "patience": 1,
        "train_epochs": 1,
        "batch_size": 16,
        "prototype_init": "randn",# "kaiming"   
        "ablation": ["no_segment","no_linear_random",'no_linear_mean',"none","no_linaer_zero","no_matching_loss","no_variances_loss",'no_prototype_layer','no_linaer']
    },
    
    
    
    
    
    
    
    
    
        "a": {
        ###
        "model_id":"mitecg",
        "data": "mitecg",
        "enc_in": 1,
        "num_class": 2,
        "seq_len":360,
        "num_prototypes":[4],#,[4],# [1,2,3,4,5],#[2],#[1,2,3,4,5], #[4],# [1,2,3,4,5],# [1,2,3,4,5], # 4
        "prototype_len": [30],#,28,30,32,36,38]   ,        #[20,25,27,30,33,35,40],#[30],# [10,20,30,40,50],#[40],#[10,20,30,40,50],# [30], # [10,20,30,40,50],# [10,20,30,40,50], # 30
        "two_class": 0,
        "is_training": 0,
        "device": "cuda:0",
        "gpu": 0,
        "train_epochs": 1,
        "model": "ProtoPTST"
    }
        
           ,
    
        "b": {
        ###
        "model_id":"mitecg",
        "data": "mitecg",
        "enc_in": 1,
        "num_class": 2,
        "seq_len":360,
        "num_prototypes":[1,2,3,4,5],#,[4],
        "prototype_len": [10,20,30,40,50],#
        "two_class": 0,
        "is_training": 0,
        "device": "cuda:2",
        "gpu": 2,
        "train_epochs": 100
    }
            ,
    
        "c": {
        ###
        "model_id":"mitecg",
        "data": "mitecg",
        "enc_in": 1,
        "num_class": 2,
        "seq_len":360,
        "num_prototypes":[1,2,3,4,5],#,[4],
        "prototype_len": [10,20,30,40,50],#
        "two_class": 0,
        "is_training": 0,
        "device": "cuda:0",
        "gpu": 0,
        "train_epochs": 1,
    },
        
        
        "saliency_mitecg": {
        ###
        "model_id":"mitecg",
        "data": "mitecg",
        "enc_in": 1,
        "num_class": 2,
        "seq_len":360,
        "num_prototypes":4,#,[4],# [1,2,3,4,5],#[2],#[1,2,3,4,5], #[4],# [1,2,3,4,5],# [1,2,3,4,5], # 4
        "prototype_len":30,#[30],# [10,20,30,40,50],#[40],#[10,20,30,40,50],# [30], # [10,20,30,40,50],# [10,20,30,40,50], # 30
        "two_class": 0,
        "patience": 1,
        "is_training":0,
        "train_epochs": 1,
        "batch_size": 16,
        "class_model_type": "lstm",
        "prototype_init": "randn",# "kaiming"   
        "ablation": ["n"]
    },
    
}





def iterate_over_parameters_two(params, iterable_key1, iterable_key2=None):
    """
    Expands a dictionary of parameters by iterating over one or two keys with iterable values.

    Args:
        params (dict): The original parameters dictionary.
        iterable_key1 (str): The first key whose values will be iterated over.
        iterable_key2 (str, optional): The second key whose values will be iterated over. Defaults to None.

    Returns:
        list: A list of parameter dictionaries with expanded values.
    """
    # Check if iterable_key1 exists and is a list
    if iterable_key1 not in params or not isinstance(params[iterable_key1], list):
        return [params]  # If the key is not iterable, return the original parameters as is.

    expanded_params = []
    
    # Case 1: Only iterate over iterable_key1
    if iterable_key2 is None:
        for value1 in params[iterable_key1]:
            param_copy = params.copy()
            param_copy[iterable_key1] = value1
            expanded_params.append(param_copy)
        return expanded_params

    # Case 2: Iterate over both iterable_key1 and iterable_key2
    if iterable_key2 not in params or not isinstance(params[iterable_key2], list):
        raise ValueError(f"The key '{iterable_key2}' must exist in params and be a list.")

    for value1 in params[iterable_key1]:
        for value2 in params[iterable_key2]:
            param_copy = params.copy()
            param_copy[iterable_key1] = value1
            param_copy[iterable_key2] = value2
            expanded_params.append(param_copy)

    return expanded_params

# Step 4: Call exp.py for each parameter set
def run_experiments(params_list, script):
    for i, params in enumerate(params_list):
        print(f"Running Experiment {i + 1} with parameters:")
        print(json.dumps(params, indent=4))

        # Build the command for exp.py
        command = ["python", script]
        for key, value in params.items():
            command.append(f"--{key}")
            command.append(str(value))

        # Call exp.py
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Experiment {i + 1} failed!")
        else:
            print(f"Experiment {i + 1} completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with parameter iterations.")
    parser.add_argument("--group", "-g", type=str, default="default", 
                       choices=parameter_groups.keys(),
                       help="Select a predefined parameter group")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--script", type=str, default="exp_saliency.py")
    parser.add_argument("--primary_param", type=str, default="num_prototypes",
                       choices=["num_prototypes", "prototype_len", "ablation"],
                       help="Primary parameter to iterate over")
    parser.add_argument("--secondary_param", type=str, default=None,
                       choices=[None, "num_prototypes", "prototype_len", "ablation"],
                       help="Optional secondary parameter to iterate over")
    
    args = parser.parse_args()
    base_params = parameter_groups.get(args.group, {}).copy()
    base_params["is_training"] = args.is_training
    
    # 验证选择的参数是否为列表类型
    selected_params = [args.primary_param]
    if args.secondary_param:
        selected_params.append(args.secondary_param)
    
    # 验证选中的参数必须是列表
    for param in selected_params:
        if param not in base_params:
            raise ValueError(f"Parameter '{param}' not found in group '{args.group}'")
        if not isinstance(base_params[param], list):
            raise ValueError(f"Parameter '{param}' in group '{args.group}' must be a list, got {type(base_params[param])}")
    
    # 验证未选中的参数不能是列表
    all_possible_params = ["num_prototypes", "prototype_len", "ablation"]
    for param in all_possible_params:
        if param in base_params and param not in selected_params:
            if isinstance(base_params[param], list):
                raise ValueError(f"Non-selected parameter '{param}' in group '{args.group}' must not be a list")
    
    all_iterations = iterate_over_parameters_two(base_params, args.primary_param, args.secondary_param)
    run_experiments(all_iterations, args.script)
    
    