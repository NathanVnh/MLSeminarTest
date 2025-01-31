# %%
import hydra
import time
from omegaconf import DictConfig
from pathlib import Path

import torch
from sound_model.inv_problems import load_chunck, generate_invp
from sound_model.metrics import SI_SDRi

from utils.im_invp_utils import InverseProblem
from posterior_samplers.diffusion_utils import load_epsilon_net

from posterior_samplers.dps import dps
from posterior_samplers.mgdm import mgdm
from posterior_samplers.reddiff import reddiff
from posterior_samplers.ddnm import ddnm_plus
from posterior_samplers.diffpir import diffpir
from posterior_samplers.daps import daps
from posterior_samplers.pnp_dm.algo import pnp_dm
from posterior_samplers.pgdm import pgdm
from local_paths import REPO_PATH

from utils.experiments_tools import (
    update_sampler_cfg,
    get_gpu_memory_consumption,
    fix_seed,
    save_track,
    save_audio,
)

fix_seed(125636)


@hydra.main(config_path="configs/exp_sound/", config_name="config")
def run_sampler(cfg: DictConfig):

    device = cfg.device
    torch.set_default_device(device)

    update_sampler_cfg(cfg)

    save_audio_path = REPO_PATH / Path(cfg.save_folder) / f"sound/{cfg.track_name}"
    save_audio_path.mkdir(parents=True, exist_ok=True)

    # load model
    epsilon_net = load_epsilon_net(cfg.dataset, cfg.sampler.nsteps, cfg.device)
    n_instruments = epsilon_net.net.n_instruments
    len_chunk = epsilon_net.net.len_chunk
    shape = (n_instruments, len_chunk)

    # create inverse prob
    x_orig = load_chunck(cfg.track_name, cfg.chunk_idx, **cfg.data)
    x_orig = x_orig.to(device)

    obs, operator, log_pot = generate_invp(
        x_orig, task=cfg.task, obs_std=cfg.std, device=device
    )

    print(f"{' Saved reference audio ':=^50}")
    save_audio(
        x_orig.sum(dim=0), "mixture_reference", save_audio_path, cfg.data.sample_rate
    )

    inverse_problem = InverseProblem(
        obs=obs, H_func=operator, std=cfg.std, log_pot=log_pot, task=cfg.task
    )

    sampler = {
        "dps": dps,
        "mgdm": mgdm,
        "diffpir": diffpir,
        "ddnm": ddnm_plus,
        "pnp_dm": pnp_dm,
        "daps": daps,
        "pgdm": pgdm,
        "reddiff": reddiff,
    }[cfg.sampler.name]

    acp_T = epsilon_net.alphas_cumprod[-1]
    initial_noise = (1 - acp_T).sqrt() * torch.randn(cfg.sampler.nsamples, *shape)
    start_time = time.perf_counter()
    samples = sampler(
        initial_noise=initial_noise,
        inverse_problem=inverse_problem,
        epsilon_net=epsilon_net,
        **cfg.sampler.parameters,
    )
    end_time = time.perf_counter()

    print(f"{' Saved mixture reconstructions ':=^50}")
    for idx, sample in enumerate(samples):
        save_audio(
            x_orig.sum(dim=0),
            f"mixture_reconstruction_{idx}",
            save_audio_path,
            cfg.data.sample_rate,
        )

    print(f"{' Reconstructions instruments ':=^50}")
    for idx, sample in enumerate(samples):
        save_chunk_path = save_audio_path / f"separation_{idx}"
        save_chunk_path.mkdir()

        save_track(sample, save_chunk_path, cfg.data.stem_names, cfg.data.sample_rate)

    si_sdri = SI_SDRi()

    print(f"{cfg.sampler} metrics")
    print("SI-SDRi:")
    for sample in samples:
        print("\t", si_sdri.score(sample, x_orig))
    print("===================")
    print(f"runtime: {end_time - start_time}")
    print(f"GPU: {get_gpu_memory_consumption(device)}")


# # for interactive window
# sys.argv = [sys.argv[0],]

if __name__ == "__main__":
    run_sampler()
