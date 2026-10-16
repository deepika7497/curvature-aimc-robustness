# Input Loss Curvature as a Predictor of Sample Vulnerability to Hardware-Noise

Code for the DATE 2026 Late Breaking Results paper by D. Sharma, D. Ravikumar, C.-H. Ho,
S. D. Kodge and K. Roy (Purdue University).
[PDF](aihwkit_experiments/paper/DATE_2026_LBR_curvature.pdf)

We show that samples with high input loss curvature are much more likely to be misclassified
when a model runs on an analog in-memory-computing (AIMC) accelerator. We test this with ADC
quantization, weight noise and PCM device noise, simulated in IBM AIHWKit.

The experiment code and full documentation are in
**[`aihwkit_experiments/README.md`](aihwkit_experiments/README.md)**.

---

## Quickstart

```bash
git clone <this-repo> && cd <this-repo>          # run every script from this root

# 1. Environment (Python 3.10, CUDA 12.1)
conda create -n aihwkit_gpu python=3.10 && conda activate aihwkit_gpu
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt                  # for GPU AIHWKit see the note in requirements.txt

# 2. Data: CIFAR-10/100 download automatically on first use into data/
python utils/download_tiny_imagenet.py           # Tiny-ImageNet into data/TinyImageNet

# 3. Reproduce the CIFAR-100 / ResNet-18 result (Fig. 1)
python aihwkit_experiments/scripts/train_model.py       --model resnet18 --dataset cifar100 --seed 42 --epochs 200 --batch_size 128 --lr 0.1
python aihwkit_experiments/scripts/compute_curvature.py --model resnet18 --dataset cifar100
python aihwkit_experiments/scripts/analog_inference2.py --model resnet18 --dataset cifar100 --seed 42 --batch_size 128
python aihwkit_experiments/scripts/analyze_curvature_failures2.py --model resnet18 --dataset cifar100 --split val --seed 42 --n_bins 10
python aihwkit_experiments/scripts/paper_figures.py     --model resnet18 --dataset cifar100
```

ImageNet uses torchvision's pretrained weights. The data path `/local/a/imagenet/imagenet2012/`
is hardcoded in the scripts, so edit it to point at your own copy.

---

## Repository layout

| Path | Contents |
|---|---|
| `aihwkit_experiments/` | Paper code: `scripts/`, `jobs/` (batch command lists), `paper/` (PDF) |
| `utils/` | Shared code for data loaders, model creation and checkpoints, curvature, evaluation and failed-index tracking, training loops |
| `models/` | CIFAR and Tiny-ImageNet ResNets, plus other architectures |
| `multirunner_aihw.sh` | Runs a job file across `host,gpu` pairs over SSH (see below) |
| `requirements.txt` | Package versions from the environment used for the paper |

Results, checkpoints, curvature scores and datasets are not in the repo. Scripts write them
to `aihwkit_experiments/results/`, `aihwkit_experiments/scores/` and `data/`.

## Running jobs across GPUs

Each file in `aihwkit_experiments/jobs/` lists one command per line. `multirunner_aihw.sh`
starts one command per `host,gpu_id` pair over SSH, sets `CUDA_VISIBLE_DEVICES`, and waits for
each wave of jobs to finish before starting the next.

```bash
cp hosts.example.txt hosts.txt                   # edit: one "host,gpu_id" per line
./multirunner_aihw.sh hosts.txt aihwkit_experiments/jobs/analog_inference2_jobs.sh
```

Before using it, edit `wd` (the repo path) and the conda setup line in the script to match
your cluster.
