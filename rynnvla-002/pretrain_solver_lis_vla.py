"""Training entry point for LIS-VLA.

Identical to ``pretrain_solver_awm_w_ck_action_head.py`` except that it builds
``ChameleonXLLMXForConditionalGeneration_LIS``. All LIS losses are returned in the
model's ``additional_loss_dict`` and are therefore summed and logged by the
unmodified RynnVLA-002 trainer.

Typical use: fine-tune from a released RynnVLA-002 VLA checkpoint (``--init_from``);
the LIS heads are initialised from scratch automatically.
"""
from accelerate import init_empty_weights
import torch

from model import ChameleonXLLMXForConditionalGeneration_LIS, LISVLAConfig, load_lis_model
from model.modeling_lis_vla import LIS_DEFAULTS
from pretrain_solver_awm_w_ck_action_head import ItemProcessor  # noqa: F401  (re-exported for the data loader)
from pretrain_solver_awm_w_ck_action_head import Solver as BaseSolver


def _str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


class Solver(BaseSolver):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        for k, v in LIS_DEFAULTS.items():
            if isinstance(v, bool):
                parser.add_argument(f"--{k}", type=_str2bool, default=v)
            else:
                parser.add_argument(f"--{k}", type=type(v), default=v)
        return parser

    def _lis_kwargs(self):
        return {k: getattr(self.args, k) for k in LIS_DEFAULTS}

    def _common_kwargs(self):
        return dict(
            action_dim=self.args.action_dim,
            time_horizon=self.args.time_horizon,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            **self._lis_kwargs(),
        )

    def _model_func(self, init_from: str):
        if self.dp_rank == 0:
            model = load_lis_model(init_from, torch_dtype=torch.bfloat16, device_map="cpu", **self._common_kwargs())
        else:
            with init_empty_weights():
                config = LISVLAConfig.from_pretrained(init_from, torch_dtype=torch.bfloat16, **self._common_kwargs())
                model = ChameleonXLLMXForConditionalGeneration_LIS(config)
        if hasattr(model.model, "vqmodel"):
            del model.model.vqmodel
        return model, None

    def _make_and_save_starting_point(self, save_path: str) -> None:
        pretrained_name = {"7B": "Alpha-VLLM/Chameleon_7B_mGPT", "34B": "Alpha-VLLM/Chameleon_34B_mGPT"}[self.args.model_size]
        model = load_lis_model(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            **self._lis_kwargs(),
        )
        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])
        model.save_pretrained(save_path, max_shard_size="10GB")


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run_with_eval_awm_w()
