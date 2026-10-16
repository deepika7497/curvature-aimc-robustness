# Input Loss Curvature vs. Analog Hardware Noise

Code and results for the DATE 2026 Late Breaking Results paper:

> **Input Loss Curvature as a Predictor of Sample Vulnerability to Hardware-Noise**
> D. Sharma, D. Ravikumar, C.-H. Ho, S. D. Kodge, K. Roy — Purdue University
> ([PDF](paper/DATE_2026_LBR_curvature.pdf))

We compute a per-sample input loss curvature score, run each model through IBM AIHWKit under
8 analog non-ideality profiles, and measure how the *analog-only* failure rate (correct in
digital FP32, wrong in analog) changes across curvature deciles.

---

## Quickstart

Run every command from the **repository root**, not from this folder. The scripts import
`utils/` and `models/` from the root, and data paths such as `data/CIFAR100` are relative to
it. For environment setup and data download, see the [root README](../README.md#quickstart).

```bash
conda activate aihwkit_gpu     # Python 3.10, torch 2.3.1, AIHWKit 0.9.2 (CUDA build)

# Reproduce the CIFAR-100 / ResNet-18 result (Fig. 1 of the paper)
python aihwkit_experiments/scripts/train_model.py        --model resnet18 --dataset cifar100 --seed 42 --epochs 200 --batch_size 128 --lr 0.1
python aihwkit_experiments/scripts/compute_curvature.py  --model resnet18 --dataset cifar100
python aihwkit_experiments/scripts/analog_inference2.py  --model resnet18 --dataset cifar100 --seed 42 --batch_size 128
python aihwkit_experiments/scripts/analyze_curvature_failures2.py --model resnet18 --dataset cifar100 --split val --seed 42 --n_bins 10
python aihwkit_experiments/scripts/paper_figures.py      --model resnet18 --dataset cifar100
```

The figure is written to `aihwkit_experiments/results/paper_figures/cifar100/resnet18/curvature_failure_rate_val_seed42.png`.

Skip `train_model.py` if a `seed42` checkpoint already exists in `results/checkpoints/`.
ImageNet always uses torchvision's pretrained `IMAGENET1K_V1` weights and is never trained.
The paper's models use seed 42. `jobs/train_model_jobs.sh` trains seeds 749 and 1005, for
checking that results hold across seeds.

**Running a whole batch across GPUs.** Each file in [`jobs/`](jobs/) lists one command per
line. `multirunner_aihw.sh` (in the root) runs those commands over SSH on a list of
`host,gpu_id` pairs (see `hosts.example.txt`):

```bash
./multirunner_aihw.sh hosts.txt aihwkit_experiments/jobs/analog_inference2_jobs.sh
```

---

## Pipeline

| Step | Script | Writes to |
|---|---|---|
| 1. Train FP models (CIFAR, Tiny-ImageNet) | `train_model.py` | `results/checkpoints/<dataset>/<model>_seed<seed>_<timestamp>/` |
| 2. Per-sample curvature | `compute_curvature.py` | `scores/<model>_<dataset>_{train,val}_curvature.npy` |
| 3. Analog inference, 8 profiles | `analog_inference2.py` | `results/analog_inference2/<dataset>/<model>/` |
| 4. Decile binning + Spearman stats | `analyze_curvature_failures2.py` | `results/basic_stats2/<dataset>/<model>/` |
| 5. Paper figure (Fig. 1) | `paper_figures.py` | `results/paper_figures/<dataset>/<model>/` (`.png` + `.csv`) |
| 6. Low/high-curvature examples (Fig. 2) | `visualize_curvature_samples.py` | `results/curvature_visualization/<dataset>/<model>/` |

Steps 3–6 find the newest `seed42` checkpoint on their own. Pass `--checkpoint <path>` to
choose a different one.

### Step details

**Curvature** (`compute_curvature.py` → `utils/curv_utils.py::get_regularized_curvature_for_batch`).
This is Hutchinson-style finite-difference curvature with Rademacher vectors, `h = 1e-3` and
`n = 10` (Eq. 1 in the paper), computed once on the final full-precision model. **The `.npy`
files store `log(curvature)`** as a dict `{sample_index: float}`. Load them with
`np.load(path, allow_pickle=True).item()`. The script exits if validation accuracy is below 50%.

**Analog inference** (`analog_inference2.py`). Every Conv and Linear layer is converted with
`convert_to_analog`. Tiles are at most 512×512, biases are digital, weights use layer-wise
symmetric remap and Gaussian clipping (σ = 2.5). `get_rpu_config([adc, wnoise, pcm])`
switches on each non-ideality:

| Flag | Setting |
|---|---|
| ADC | 9-bit output resolution, `out_bound = 12`, `ABS_MAX` noise management |
| WNoise | `ADDITIVE_CONSTANT` weight noise, `w_noise = 0.02` |
| PCM | `PCMLikeNoiseModel(g_max=25, prog/read/drift scale = 1.0)` |

The 8 profiles are `ideal, only_ADC, only_wnoise, only_PCM, ADC_and_wnoise, ADC_and_PCM,
wnoise_and_PCM, ADC_wnoise_and_PCM`. The script evaluates the train and val splits for each
profile and saves:

- `failed_indices_{train,val}_{fp,<profile>}.pkl`: sets of misclassified sample indices
- `profile_summary_<model>_<dataset>.txt`: accuracy table for all profiles
- `<model>_analog_<profile>_<timestamp>.pth`: the programmed analog model, about 90–200 MB each

**Analysis** (`analyze_curvature_failures2.py`). Sorts samples by curvature into `--n_bins`
equal-size bins. For each bin, it computes the analog-only failure rate as
`|analog fails ∧ FP correct| / |FP correct|`, plus the Spearman correlation between curvature
and failure. Results go to `statistics_<profile>_<split>_seed42.txt`. Table I in the paper
reports bins D1 and D10 of the `ADC_wnoise_and_PCM` profile.

---

## Folder layout

```
aihwkit_experiments/
├── scripts/            # Python entry points (see Pipeline)
│   └── plot_motivation_accuracy_vs_noise.py   # FP vs. analog accuracy bar chart (poster)
├── jobs/               # one-command-per-line job lists for multirunner_aihw.sh
├── paper/              # paper PDF
│
│   # created when you run the scripts (git-ignored):
├── scores/             # curvature .npy files
├── logs/
└── results/
    ├── checkpoints/            # FP checkpoints (one per epoch, can reach 100+ GB)
    ├── analog_inference2/      # failed indices + programmed analog models
    ├── basic_stats2/           # decile statistics
    ├── paper_figures/          # Fig. 1 (+ CSV of the plotted numbers)
    └── curvature_visualization/# Fig. 2
```

The `2` suffix (`analog_inference2.py`, `analyze_curvature_failures2.py`) marks the version
used for the paper. An earlier version swept PCM drift time and is not in this repo.

The scripts depend on `../utils/`, `../models/` and `../data/`. Importing `utils` tries to
load `utils/quant_utils.py`, which needs `libraries/BRECQ`. Without it the import is skipped,
and nothing in this folder uses it. ImageNet is read from `/local/a/imagenet/imagenet2012/`,
which is hardcoded in each script.

---

## Known issues

- **The PCM profile applies programming noise only. Drift and read noise are not applied.**
  `analog_inference2.py` calls `program_analog_weights()`, which applies PCM programming noise.
  That is why `only_PCM` lowers accuracy and shows its own curve in Fig. 1. The script never
  calls `drift_analog_weights(t)`, and in AIHWKit drift and read noise are applied only by
  that call (`PCMLikeNoiseModel.apply_drift_noise_to_conductance`). So `drift_scale` and
  `read_noise_scale` have no effect, even though the paper says the PCM profile includes all
  three. To add them, call `analog_model.drift_analog_weights(t)` after programming.
- **Each profile uses one noise sample.** Weights are programmed once per profile with
  `seed=42`, so each sample's pass/fail result comes from a single random draw.
- **`--use_calibration` has no effect.** The flag is accepted, but `calibrate_input_ranges` is
  never called.
- **Training saves every epoch.** `train_model.py` writes a checkpoint each epoch. Only
  `*_best.pth` is used afterwards, so delete the rest if disk space is tight.
