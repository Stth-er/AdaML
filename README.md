# AdaML: Adaptive meta-learning for generalizable sparse-view CT reconstruction

**Guodong Li, Tianli Liao, Kun Shang, and Ce Wang**

*Computerized Medical Imaging and Graphics*

This repository provides the implementation of StaML and AdaML for sparse-view CT reconstruction. StaML learns a transferable initialization, and AdaML adapts the reconstruction model using a learnable inner-loop learning rate. The repository also includes a baseline reconstruction model.

## Installation

The AdaML environment uses Python 3.8.15, PyTorch 2.4.1 with CUDA 12.4, ODL 0.8.1, and ASTRA Toolbox 2.2.0. The CT operator uses ASTRA's CUDA backend and requires an NVIDIA GPU. Select a CUDA-enabled PyTorch build compatible with your GPU and driver; the example below uses CUDA 12.4.

```bash
conda create -n AdaML python=3.8.15 -y
conda activate AdaML

# CUDA-enabled PyTorch; select another build if needed for your system.
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies and versions observed in the AdaML environment; OpenCV is specified as the corresponding pip package. The original ASTRA installation also includes Conda CUDA libraries (CUDA toolkit 11.3), so ASTRA setup can depend on the local driver and platform. Consult the [ASTRA installation instructions](https://astra-toolbox.com/docs/install.html) if its CUDA backend is unavailable. These installation steps have not yet been verified in a fresh environment. The code explicitly uses CUDA for CT operators, so setting the PyTorch device to CPU is insufficient for CPU-only execution.

## Data Preparation

Please refer to the paper for dataset descriptions and experimental preparation. The code expects DICOM slices arranged in separate meta-training and meta-test directories, with one directory per class:

```text
DATA_ROOT/
├── meta_train/
│   ├── A/
│   │   ├── A001.dcm
│   │   └── A002.dcm
│   └── B/
│       ├── B001.dcm
│       └── B002.dcm
└── meta_test_1/
    ├── A/
    │   ├── A003.dcm
    │   └── A004.dcm
    └── B/
        ├── B003.dcm
        └── B004.dcm
```

The names above illustrate the required layout, not the paper's actual dataset split. Each slice filename must start with the name of its one-character parent directory, since the episodic loaders construct paths from the first character of the filename. Each class needs enough slices for the selected support and query sets; the test loader samples across episodes without replacement.

Set `train1` and `test1` in `main_StaML.py` to your meta-training and meta-test directories. For the baseline, update the corresponding paths in `main.py`. Both directories are initialized even during evaluation. Dataset files are not included in this repository.

## Training

AdaML and StaML share the same StaML training stage. From the repository root, update the paths above. In `main_StaML.py`, uncomment `trainer.train_StaML()` and set `inference_StaML = False`, then train using StaML mode:

```bash
# Train the shared StaML initialization
python main_StaML.py --method StaML --device cuda:0
```

The shared training checkpoint is saved under `results/models_StaML/` and used for both StaML and AdaML evaluation. `Trainer()` attempts to resume from its configured checkpoint by default. To start afresh, use `Trainer(is_restart=True)`; this also deletes previously generated visualization outputs. Other settings, including support/query sizes, learning rates, and update steps, are available through `python main_StaML.py --help`.

## Evaluation

Download the [abdomen evaluation checkpoint](https://github.com/Stth-er/AdaML/releases/download/v1.0/checkpoint_abdomen.pth) for testing on abdominal CT data. Place it at `results/models_StaML/StaML_epoch_49iter18999.pth.tar` by renaming the downloaded `checkpoint_abdomen.pth` file. The same checkpoint is used for StaML and AdaML evaluation.

Set `inference_StaML = True` and leave the training call commented out in `main_StaML.py`. The default StaML evaluation uses the checkpoint in `results/models_StaML/`:

```bash
python main_StaML.py --method StaML --device cuda:0
```

For AdaML, use the same StaML training checkpoint and select AdaML mode for test-time adaptation. Check the AdaML branch's `pth` argument in `main_StaML.py` and adjust the test settings, including `--k_spt`, `--k_qry`, `--update_step_test`, and `--hyper_lr`, for the experiment:

```bash
python main_StaML.py --method AdaML --device cuda:0
```

Both evaluation branches currently point to the same StaML checkpoint. Edit the `pth` argument in the script to select another shared checkpoint. The initial adaptation learning rate is loaded from the checkpoint; `--hyper_lr` controls the learning-rate update in AdaML mode. The script reports average PSNR, SSIM, and RMSE across adaptation steps. 



## Citation

If you use this work, please cite:

```bibtex
@article{li2026adaml,
  title   = {AdaML: Adaptive meta-learning for generalizable sparse-view CT reconstruction},
  author  = {Li, Guodong and Liao, Tianli and Shang, Kun and Wang, Ce},
  journal = {Computerized Medical Imaging and Graphics},
  year    = {2026}
}
```
