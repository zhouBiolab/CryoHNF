# CryoHNF: Structure-informed neural fields enable high-fidelity cryo-EM reconstruction

CryoHNF is a hybrid neural-field framework for cryo-EM reconstruction directly in 3D real space.
It couples a coordinate-based neural field with a pretrained structure-informed density prior
(a 3D U-Net trained on density maps simulated from PDB atomic structures) under a
differentiable physics-based forward model (projection + CTF), optimized with an image-domain
data-fidelity loss and a structural-consistency loss (default weight 1e-3). Homogeneous
reconstruction uses `--zdim 0`; heterogeneous reconstruction conditions the neural field on
per-particle latent embeddings.

## Installation

```
# conda environment (Python 3.9, PyTorch 1.13, CUDA 11.6)
conda env create -f environment.yml
conda activate cryohnf

# or with pip
pip install -r requirements.txt
```

Notes:

- PyTorch is tested with versions 1.13 / 2.0 / 2.1. Install a CUDA build matching your driver.
- `tinycudann` (tiny-cuda-nn) provides the default hash-grid positional encoding (`--pe-type
  hash_grid`). The PyPI package compiles from source and requires the CUDA toolkit (nvcc) matching
  your PyTorch CUDA build. If you cannot build it, use `--pe-type geom_lowf` instead.

## Pretrained structure-informed prior

A pretrained prior network is included and loaded automatically at startup (keep it in the
repository root next to `main.py`):

| File | Variant | Selected by |
|---|---|---|
| `model.pth` | SCUNet prior | `--refine_net scunet` |

## Quickstart

### 1. Data preprocessing

Perform a **homogeneous (consensus) refinement** in cryoSPARC to obtain particle poses and CTF
parameters: 1) import the particles, 2) run an ab initio reconstruction job, 3) run a homogeneous
refinement job, all with default parameters.

CryoHNF extracts poses and CTF parameters directly from the cryoSPARC `.cs` metadata file
(or from a RELION `.star` file).

### 2. Homogeneous reconstruction

Split the particles into two independent half-sets inside cryoSPARC (or use `--split` with a
`.star`/`.cs` file carrying random-subset flags) and train one model per half-map for
gold-standard FSC evaluation:

```
python main.py particles.cs -o out/lsu --zdim 0 --split 1   # writes to out/lsu/halfA/
python main.py particles.cs -o out/lsu --zdim 0 --split 2   # writes to out/lsu/halfB/
```

An optional loose spherical mask can speed up training: `--mask mask.mrc`.

### 3. Heterogeneous reconstruction

Train on the full particle stack with a latent embedding per particle to capture conformational
or compositional variability:

```
python main.py particles.cs -o out/spliceosome --zdim 8 -n 20
```

After training, per-particle latent embeddings are saved to `z.<epoch>.pkl` (`z_mu`, `z_logvar`,
`ind`) for downstream analysis — e.g. UMAP/PCA visualization and clustering (HDBSCAN, K-means),
then decoding cluster-representative embeddings into 3D density maps.

> **Note on batch size and GPU memory:** the paper's settings (and the examples above) assume a
> full 80 GB GPU (NVIDIA A800). On smaller GPUs, reduce `-b`: with a 40 GB GPU use `-b 4` or less
> for 128-pixel boxes and `-b 2` or less for 192-pixel boxes. Homogeneous runs (`--zdim 0`) need
> more memory than heterogeneous ones, because the MLP is evaluated over the full 3D volume.

### Outputs

- `vol_epochXXX.recon.<D>.mrc`, `vol_epochXXX.recon_mlp.<D>.mrc` — reconstructed volumes
  (view in UCSF ChimeraX)
- `weights.<epoch>.pkl` — model + optimizer checkpoints (resume with `--load`)
- `z.<epoch>.pkl` — latent embeddings (heterogeneous mode)
- `run.log`, plus intermediate reconstruction dumps every 300 iterations under the output
  directory for monitoring

### Key options

| Option | Default | Description |
|---|---|---|
| `--zdim` | required | Latent dimension; `0` = homogeneous reconstruction |
| `-n`, `--num-epochs` | 20 | Training epochs |
| `-b`, `--batch-size` | 4 | Minibatch size |
| `--poses` / `--ctf` | auto | Explicit pose/CTF `.pkl` files; otherwise parsed from `.cs`/`.star` |
| `--domain` | real | Reconstruction domain (`real`, `hartley`, `all`) |
| `--refine_net` | scunet | Structure-informed prior network (`scunet`) |
| `--pe-type` | hash_grid | Positional encoding (`hash_grid`, `geom_lowf`, `gaussian`, ...) |
| `--pe-dim` | 32 | Number of frequency bands in the positional encoding |
| `--prior-weight` | 1e-3 | Weight of the structural-consistency (prior) loss |
| `--beta` | 0 | KLD weight (encoder-based variational variant) |
| `--lazy` | off | Lazy particle loading for large datasets |
| `--load` | — | Initialize from a checkpoint |

### GPU memory and batch size

Defaults follow the paper's setup on a full 80 GB A800. On smaller GPUs, reduce `-b` and
halve it again if an out-of-memory failure appears. Verified reference points on a 40 GB
A800 MIG slice: box size 128 runs at `-b 8` (heterogeneous) / `-b 4` (homogeneous), box
size 192 at `-b 2` (homogeneous). Note that on MIG instances an out-of-memory failure can
surface as `NVML_INTERNAL_ASSERT` from the PyTorch caching allocator instead of a plain
CUDA OOM message.

## Evaluation

Gold-standard FSC, local resolution and map-model FSC in the paper were computed with
[PHENIX](https://phenix-online.org) (unmasked half-map FSC at the 0.143 criterion,
`phenix.local_resolution`) and visualized in [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/);
directional isotropy was assessed with 3D-FSC / Fourier shell occupancy analysis.

## Data availability

All datasets are public in the [EMPIAR](https://www.ebi.ac.uk/empiar/) archive:

EMPIAR-10076 (bL17-depleted LSU assembly intermediates), EMPIAR-10049 (RAG1–RAG2 signal-end
complex), EMPIAR-10028 (P. falciparum 80S ribosome), EMPIAR-10492 (SARS-CoV-2 spike protein),
EMPIAR-10180 (pre-catalytic spliceosome).

As a reference, reconstructing 10,000 particles at a 256-pixel box size takes ~4 h on one
NVIDIA A800 GPU.
