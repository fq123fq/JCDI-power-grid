This is the repository for the preprint [Diffusion Model-based Parameter Estimation in Dynamic Power Systems](https://arxiv.org/abs/2411.10431)

## Install the requiered libraries 
```
conda env create -f JCDI_env.yaml
```

## Datasets
* Dataset in our experiment can be downloaded at [this link](https://drive.google.com/drive/folders/1_ntBnNasDw7eaLQs3zlGaPJ9-GpB4BFa?usp=sharing).
* Put the data files in the folder "data_set_inverse“.

## Train JCDI
```
python train_fq_inverse_JCDI_MEL_pre_release.py -c config_inverse/config_inverse_JCDI_MEL_pre_release.json
```

## Please cite the preprint if you find this repository helpful
```
@misc{zhu2025diffusionmodelbasedparameterestimation,
      title={Diffusion Model-based Parameter Estimation in Dynamic Power Systems}, 
      author={Feiqin Zhu and Dmitrii Torbunov and Zhongjing Jiang and Tianqiao Zhao and Amirthagunaraj Yogarathnam and Yihui Ren and Meng Yue},
      year={2025},
      eprint={2411.10431},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2411.10431}, 
}
```

## Acknowledgement

Our code for the proposed model builds upon the diffusion model implementation provided by [SSSD](https://github.com/AI4HealthUOL/SSSD).
We gratefully acknowledge the authors for releasing and maintaining their source code.

