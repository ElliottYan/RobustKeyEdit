from dataclasses import dataclass
from typing import List, Literal

from ...util.hparams import HyperParams
import yaml


@dataclass
class EMMETHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    alg_name: str
    device: int
    model_name: str
    stats_dir: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False

    update_norm_lambda: float = 0
    emmet_lambda: float = 0.1

    # Aggregator
    use_lora_aggregator: bool = False
    use_our_aggregator: bool = False
    num_train_agg_steps: int = 10
    agg_lora_rank: int = 8
    
    agg_norm_c: bool = True
    agg_remove_c: bool = True
    agg_loss_weight: float = 2.0
    agg_loss_norm: bool = True
    kl_loss_weight: float = 1.0
    spread_loss_weight: float = 1.0
    kl_loss_type: str = 'whiten_diff'
    agg_use_cls: bool = False
    target_consistency_weight: float = 1.0
    cls_loss_weight: float = 0.0
    cls_threshold: float = 0.9
    agg_train_on_held: bool = False
    agg_lr: float = 5e-4
    agg_gate_inner_dim_ratio: float = 0.1

    # data
    agg_add_ori: bool = True
    agg_half_reph_subj: bool = True
    agg_batch_size: int = 15
    agg_add_rephrases: bool = False
    agg_add_para_attack: bool = False
    agg_add_long_context: bool = False
    agg_add_shuffle: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'EMMET') or print(f'EMMETHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
