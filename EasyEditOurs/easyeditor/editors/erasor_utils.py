import torch
import copy

def compare_models(model_a, model_b):
    """
    Compare the weights of two PyTorch models and store the difference from model A to model B.
    
    Args:
    model_a (nn.Module): The first PyTorch model
    model_b (nn.Module): The second PyTorch model
    
    Returns:
    dict: A dictionary where the key is the model's module name and the value is the weight difference
    """
    difference = {}
    b_para_dict = {n:v for n, v in model_b.named_paramters()}
    
    for (name_a, param_a) in model_a.named_parameters():
        if name_a not in b_para_dict:
            difference[name_a] = param_a.detach().cpu()
        else:
            param_b = b_para_dict[name_a]
            if param_a.shape != param_b.shape:
                raise ValueError(f"Parameter shapes do not match for {name_a}: {param_a.shape} vs {param_b.shape}")
            diff = param_b.data - param_a.data
            difference[name_a] = diff.detach().cpu()
    
    return difference

def restore_model(model_a, difference):
    """
    Restore model B given model A and the difference dictionary.
    
    Args:
    model_a (nn.Module): The base PyTorch model (like model A)
    difference (dict): The dictionary containing the weight differences from A to B
    
    Returns:
    nn.Module: The restored PyTorch model (should be like model B)
    """
    restored_model = copy.deepcopy(model_a)
    
    for name, param in restored_model.named_parameters():
        if name in difference:
            diff = difference[name]
            if isinstance(diff, torch.Tensor):
                # If the difference is a tensor, add it to the parameter
                param.data += diff
            else:
                # If the difference is the entire parameter (for parameters not in model B),
                # replace the parameter with the stored value
                param.data = diff.data
    
    # Handle parameters in difference that are not in model_a (new parameters in B)
    for name, diff in difference.items():
        if name not in dict(restored_model.named_parameters()):
            # This is a new parameter in B that doesn't exist in A
            # We need to add this parameter to the restored model
            module_name, param_name = name.rsplit('.', 1)
            module = restored_model.get_submodule(module_name)
            setattr(module, param_name, torch.nn.Parameter(diff))
    
    return restored_model
