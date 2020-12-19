# GraspNet Baseline
Baseline model for [GraspNet-1Billion](https://graspnet.net/) dataset (CVPR 2020).

[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf)][[dataset](https://graspnet.net/)][[API](https://github.com/graspnet/graspnetAPI)][[doc](https://graspnetapi.readthedocs.io/en/latest/index.html)]

![teaser](doc/teaser.png)

## Requirements
- Python 3
- PyTorch 1.6
- TensorBoard 2.3
- NumPy
- SciPy
- Pillow
- tqdm

## Installation
Get the code.
```bash
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators. (Code adapted from [votenet](https://github.com/facebookresearch/votenet))
```bash
cd pointnet2
python setup.py install
cd ..
```
Compile and install knn operator. (Code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda))
```bash
cd knn
python setup.py install
cd ..
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
cd ..
```

## Tolerance Label Generation
Tolerance labels are not included in the original dataset, and need additional generation. The generation code is in [`dataset/generate_tolerance_label.py`](dataset/generate_tolerance_label.py). You can simply generate tolerance label by running the scripts: (`--dataset_root` should be specified according to your settings)
```bash
cd dataset
sh command_generate_tolerance_label.sh
cd ..
```

## Training and Testing
Training examples are shown in [command_train.sh](command_train.sh). `--dataset_root`, `--camera` and `--log_dir` should be specified according to your settings. You can use TensorBoard to visualize training process.

Testing examples are shown in [command_test.sh](command_test.sh), which contains inference and result evaluation. `--dataset_root`, `--camera`, `checkpoint_path` and `--dump_dir` should be specified according to your settings.


