# Data generation module applied in the paper: Benchmarking Safety Monitors for Image Classifiers with Machine Learning
A python repository that creates 79 benchmark datasets containing five categories of out-of-distribution data for image classifiers: class novelty, noise, anomalies,  distributional shifts, and adversarial attacks.

Main repository at: https://github.com/raulsenaferreira/PRDC_2021_SUT_module

A module applied to generate tables and visualize results can be found in another repository: https://github.com/raulsenaferreira/PRDC_2021_Evaluation_module

## If you use this repository or part of it please consider to cite the respective paper
```
@article{ferreira2021benchmarking,
  title={Benchmarking Safety Monitors for Image Classifiers with Machine Learning},
  author={Ferreira, Raul Sena and Arlat, Jean and Guiochet, J{\'e}r{\'e}mie and Waeselynck, H{\'e}l{\`e}ne},
  journal={26th IEEE Pacific Rim International Symposium on Dependable Computing (PRDC 2021), Perth, Australia},
  year={2021}
}
```

## Simple installing

python -m venv env

.\env\Scripts\activate

pip install -r requirements.txt

## Usage

python main.py threat_type 1 0 0 path_for_saving_data
