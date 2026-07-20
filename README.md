<h1 align="center">
  EdMCGS: Event-driven Markov Chain Gaussian Splatting for Extreme-Low-Frame-Rate Dynamic Scene Reconstruction
</h1>

## Datasets

Our synthesis dataset: https://huggingface.co/datasets/JosephEclipse/EdMCGS

## Run

```bash
# Our synthesis datasets
python train_gui.py -s ../PATH_TO_SYNTHESIS_DATASET/kick_5fps -m ../OUTPUT_PATH --eval --is_blender --iterations 30000
```

```bash
# Event-boosted D3DGS real-world dataset
python train_gui.py -s ../PATH_TO_YOUR_DATASET/SunFlowers -m ../OUTPUT_PATH --eval --iterations 40000
```

## Acknowledgement

```
@article{kerbl20233d,
  title={3d gaussian splatting for real-time radiance field rendering.},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Trans. Graph.},
  volume={42},
  number={4},
  pages={139--1},
  year={2023}
}
```

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```
