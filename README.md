# JCDI: Joint Conditional Diffusion Model-based Inverse problem Solver
This is the repository for the preprint [Diffusion Model-based Parameter Estimation in Dynamic Power Systems](https://arxiv.org/abs/2411.10431).

**Authors:**
Feiqin Zhu\*, 
Dmitrii Torbunov\*, 
Zhongjing Jiang, 
Tianqiao Zhao, 
Amirthagunaraj Yogarathnam, 
Yihui Ren†, 
Meng Yue†

\* equal contribution, † supervisors

[[`Paper`](https://arxiv.org/abs/2411.10431)]
[[`BibTeX`](#citing-JCDI)]

![JCDI-framework](/assets/JCDI-framework.jpg "JCDI-framework")

We present a novel probabilistic parameter estimation framework based on the generative diffusion model, named *Joint Conditional Diffusion Model-based Inverse Problem Solver* (JCDI). 
In this framework, we train the conditional diffusion model to learn the inherent distributions within parameter space and provide a data-driven solution for parameter estimation.
We condition the diffusion model on system observations and generate system parameters that are constrained by the observations.
To address the parameter non-uniqueness problem, we introduce the multi-event joint conditioning mechanism: *conditioning the diffusion model on multiple observations under various disturbances*. 
Successful verification of JCDI has been achieved for composite load model parameterization in power systems.  

## Install the requiered libraries 
```
conda env create -f JCDI_env.yml
```

## Datasets
* Dataset in our experiment can be downloaded at [this link](https://drive.google.com/drive/folders/1_ntBnNasDw7eaLQs3zlGaPJ9-GpB4BFa?usp=sharing).
* Put the data files in the folder "data_set_inverse“.

## Train JCDI
```
python main.py -c config_inverse/config_JCDI.json
```

## Citing JCDI
Please cite the preprint if you find this repository helpful.
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
This work was supported by the Advanced Grid Modeling Program, Office of Electricity of the U.S. Department of Energy under Agreement 39917.

Our code for the proposed model builds upon the diffusion model implementation provided by [SSSD](https://github.com/AI4HealthUOL/SSSD).
We gratefully acknowledge the authors for releasing and maintaining their source code.

