"""Mock end-to-end test of tools/branch_horizons.py (no simulator, no 7B model).

A fake environment succeeds once the summed first action dimension reaches a threshold; System 1 always
outputs a slightly negative action (so it fails), proposer candidates are random. Longer intervention
horizons should therefore rescue more failure points - the test checks the bookkeeping, not the claim.
"""
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
H = 5


class FakeEnv:
    horizon = 120

    def __init__(self):
        self.reset()

    def reset(self):
        self.x, self.steps, self.done = 0.0, 0, False
        return self.obs()

    def obs(self):
        img = np.zeros((8, 8, 3), np.uint8)
        return {"agentview_image": img, "robot0_eye_in_hand_image": img, "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0, 0, 0, 1.0]), "robot0_gripper_qpos": np.zeros(2)}

    def get_sim_state(self):
        return np.array([self.x, self.steps], dtype=np.float64)

    def set_init_state(self, s):
        s = np.asarray(s)
        if s.shape == (2,):
            self.x, self.steps = float(s[0]), int(s[1])
        return self.obs()

    def step(self, a):
        if self.done:
            raise ValueError("executing action in terminated episode")
        self.x += a[0]
        self.steps += 1
        succ = self.x >= 3.0
        self.done = succ or self.steps >= self.horizon
        return self.obs(), 0.0, succ, {}

    def close(self):
        pass


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(max_position_embeddings=64)
        self.last_search_info = None
        self.g = torch.Generator().manual_seed(0)

    @property
    def device(self):
        return torch.device("cpu")

    def generate_lis_action(self, ids, gen, num_candidates, enable_search, return_all):
        prior = torch.full((H, 7), -0.05)
        if enable_search:
            cands = torch.rand((num_candidates, H, 7), generator=self.g) * 0.4 - 0.1
            cands[0] = prior
            self.last_search_info = {"search": True, "prior": prior, "candidates": cands}
        else:
            self.last_search_info = {"search": False}
        return prior


def install_stubs():
    def mod(name, **kw):
        m = types.ModuleType(name)
        m.__dict__.update(kw)
        sys.modules[name] = m
        return m

    class Bench:
        n_tasks = 2

        def get_task_init_states(self, i):
            return [np.array([0.0, 0])] * 5

        def get_task(self, i):
            return types.SimpleNamespace(language="do it")

    mod("libero"); mod("libero.libero", benchmark=types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_goal": Bench}))
    mod("libero_util")
    mod("libero_util.libero_utils", get_libero_dummy_action=lambda: [0.0] * 7,
        get_libero_env=lambda task, resolution: (FakeEnv(), None),
        get_libero_image=lambda obs, res, key="agentview_image": obs[key],
        quat2axisangle=lambda q: np.zeros(3))
    mod("libero_util.lis_utils", build_prompt_tokens=lambda *a, **k: [1, 2, 3])
    mod("data"); mod("data.pre_tokenize_action_state", ItemProcessor=lambda **k: None)
    mod("eval_lis_libero", MAX_STEPS={"libero_goal": 100}, unnorm_action=lambda a: np.asarray(a))
    mod("collect_rollouts", perturb_robot=lambda env, sigma, rng: None)
    mod("model", load_lis_model=lambda *a, **k: FakeModel())


STUBBED = ["libero", "libero.libero", "libero_util", "libero_util.libero_utils", "libero_util.lis_utils", "data",
           "data.pre_tokenize_action_state", "eval_lis_libero", "collect_rollouts", "model", "branch_horizons"]


def test_branch_horizons_end_to_end():
    saved = {k: sys.modules.get(k) for k in STUBBED}
    try:
        _run()
    finally:  # do not leak the stubs into other tests in the same session
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _run():
    install_stubs()
    import branch_horizons
    import branch_horizons_summary

    with tempfile.TemporaryDirectory() as d:
        ids = os.path.join(d, "ids.txt")
        open(ids, "w").write("100000\n100001\n100002\n")
        branch_horizons.main(["--ckpt", "x", "--lis_weights", "x", "--tokenizer_path", "x", "--out", d,
                              "--episodes", "4", "--only_ids", ids, "--horizons", "1,2,-1", "--alts", "3",
                              "--points", "2", "--every", "2", "--device", "cpu"])
        logs = [json.loads(l) for l in open(os.path.join(d, "log_shard0.jsonl"))]
        assert len(logs) == 3 and all(r["error"] is None for r in logs), logs
        e = np.load(os.path.join(d, "ep100000.npz"))
        assert e["alt_success"].shape == (2, 3, 3)
        assert not bool(e["main_success"]) and not e["s1_success"].any()
        # horizon 1 executes exactly one alternative decision; 'all' executes at least as many as horizon 2
        assert (e["n_alt"][:, 0] == 1).all()
        assert (e["n_alt"][:, 1] <= 2).all() and (e["n_alt"][:, 2] >= e["n_alt"][:, 1]).all()
        rep = branch_horizons_summary.summarise(d, base_success=(5, 10))
        assert set(rep["by_horizon"]) == {"1", "2", "all"}
        # with this toy dynamics, letting the proposer act longer must rescue at least as often
        k = lambda h: int(rep["by_horizon"][h]["rescuable_points"].split("/")[0])
        assert k("all") >= k("1")


if __name__ == "__main__":
    test_branch_horizons_end_to_end()
    print("ok")
