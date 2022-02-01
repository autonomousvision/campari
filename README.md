# CAMPARI: Camera-Aware Decomposed Generative Neural Radiance Fields
## [Paper](http://www.cvlibs.net/publications/Niemeyer2021CVPR.pdf) | [Supplementary](http://www.cvlibs.net/publications/Niemeyer2021CVPR_supplementary.pdf) | [Video](http://www.youtube.com/watch?v=rrIIEc2qYjM&vq=hd1080&autoplay=1) | [Poster](http://www.cvlibs.net/publications/Niemeyer2021THREEDV_poster.pdf)

If you find our code or paper useful, please cite as

    @inproceedings{CAMPARINiemeyer2021,
        author = {Niemeyer, Michael and Geiger, Andreas},
        title = {CAMPARI: Camera-Aware Decomposed Generative Neural Radiance Fields},
        booktitle = {International Conference on 3D Vision (3DV)},
        year = {2021}
    }

## TL; DR - Quick Start

![Faces](vis.gif)

First you have to make sure that you have all dependencies in place. The simplest way to do so, is to use [anaconda](https://www.anaconda.com/).

You can create an anaconda environment called `campari` using
```
conda env create -f environment.yml
conda activate campari
```

You can now test our code on the provided pre-trained models.
For example, for creating short video clips, run simply run
```
python eval_video.py configs/celeba_pretrained.yaml
```
or
```
python eval_figures.py configs/celeba_pretrained.yaml
```
for creating respective figures.

This script should create a model output folder `out/celeba_pretrained`.
The animations are then saved to the respective subfolders.

## Usage

### Datasets and Stats Files

To train a model from scratch, you have to download the respective dataset.

For this, please run
```
bash scripts/download_dataset.sh
```
and following the instructions. This script should download and unpack the data automatically into the `data/` folder.


**Note:** For FID evaluation or creating figures containing the GT camera distributions, you need to download the "stats files" (select "4 - Camera stats files" for this).


### Controllable Image Synthesis

To render short clips or figures from a trained model, run
```
python eval_video.py CONFIG.yaml
```
or
```
python eval_figures.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.
The easiest way is to use a pre-trained model.
You can do this by using one of the config files which are indicated with `*_pretrained.yaml`. 

For example, for our model trained on celebA, run
```
python eval_video.py configs/celeba_pretrained.yaml
```
Our script will automatically download the model checkpoints and render images.
You can find the outputs in the `out/*_pretrained` folders.

Please note that the config files  `*_pretrained.yaml` are only for evaluation or rendering, not for training new models: when these configs are used for training, the model will be trained from scratch, but during inference our code will still use the pre-trained model.

### FID Evaluation
For evaluation of the models, we provide the script `eval_fid.py`. Make sure to have downloaded the stats files (see Usage - Datasets and Stats Files). You can run it using
```
python eval_fid.py CONFIG.yaml
```
The script generates 20000 images and calculates the FID score.

### Training
Finally, to train a new network from scratch, run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./logs
```
where you replace `OUTPUT_DIR` with the respective output directory. For available training options, please take a look at `configs/default.yaml`.

# Futher Information

## More Work on Coordinate-based Neural Representations
If you like the CAMPARI project, please check out related works on neural representions from our group:
- [Niemeyer et. al. - GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields (CVPR'21)](https://github.com/autonomousvision/giraffe)
- [Schwarz et. al. - GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis (NeurIPS'20)](https://github.com/autonomousvision/graf)
- [Niemeyer et. al. - DVR: Learning Implicit 3D Representations without 3D Supervision (CVPR'20)](https://github.com/autonomousvision/differentiable_volumetric_rendering)
- [Oechsle et. al. - Learning Implicit Surface Light Fields (3DV'20)](https://arxiv.org/abs/2003.12406)
- [Peng et. al. - Convolutional Occupancy Networks (ECCV'20)](https://arxiv.org/abs/2003.04618)
- [Niemeyer et. al. - Occupancy Flow: 4D Reconstruction by Learning Particle Dynamics (ICCV'19)](https://avg.is.tuebingen.mpg.de/publications/niemeyer2019iccv)
- [Oechsle et. al. - Texture Fields: Learning Texture Representations in Function Space (ICCV'19)](https://avg.is.tuebingen.mpg.de/publications/oechsle2019iccv)
- [Mescheder et. al. - Occupancy Networks: Learning 3D Reconstruction in Function Space (CVPR'19)](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks)
