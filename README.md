# JCDI: Joint Conditional Diffusion Model-based Inverse Problem Solver
This is the repository for the paper [Diffusion Model-based Parameter Estimation in Dynamic Power Systems](https://www.nature.com/articles/s44172-026-00670-z), which has been published in *Communications Engineering*.

**Authors:**
Feiqin Zhu\*, 
Dmitrii Torbunov\*, 
Zhongjing Jiang, 
Tianqiao Zhao, 
Amirthagunaraj Yogarathnam, 
Yihui Ren†, 
Meng Yue†

\* equal contribution, † supervisors

[[`Paper`](https://www.nature.com/articles/s44172-026-00670-z)]
[[`BibTeX`](#citing-JCDI)]

![JCDI-framework](/assets/JCDI-framework-update.jpg "JCDI-framework")

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
* Datasets for this work, including datasets for training and evaluation, trained model checkpoints, and source data for graphs and charts, have been deposited in the Zenodo Repository: [JCDI dataset](https://zenodo.org/records/18980716).
* To train the model, create a new folder named "data_set_inverse" and put the data files for training and evaluation in the folder.

## Train JCDI
```
python main.py -c config_inverse/config_JCDI.json
```

## Overall flowchart of JCDI implementation
In addition to training, the overall flowchart of JCDI implementation is shown below. It includes six modules: sensitivity analysis, data generation, model training, parameter inference, dynamic response prediction, and result visualization & analysis.

![JCDI-flowchart](/assets/JCDI-flowchart.jpg "JCDI-flowchart")

## Citing JCDI
If you find this repository helpful, please cite the paper using the following BibTeX entry.
```
@article{zhu2026diffusion,
  title={Diffusion model-based parameter estimation in dynamic power systems},
  author={Zhu, Feiqin and Torbunov, Dmitrii and Jiang, Zhongjing and Zhao, Tianqiao and Yogarathnam, Amirthagunaraj and Ren, Yihui and Yue, Meng},
  journal={Communications Engineering},
  year={2026},
  doi={https://doi.org/10.1038/s44172-026-00670-z}
}
```

## Acknowledgement
This work was supported by the Advanced Grid Modeling Program, Office of Electricity of the U.S. Department of Energy under Agreement 39917.

Our code for the proposed model builds upon the diffusion model implementation provided by [SSSD](https://github.com/AI4HealthUOL/SSSD).
We gratefully acknowledge the authors for releasing and maintaining their source code.

