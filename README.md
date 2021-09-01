# Data generation module applied in the paper: Benchmarking Safety Monitors for Image Classifiers with Machine Learning
A python repository that creates 79 benchmark datasets containing five categories of out-of-distribution data for image classifiers: class novelty, noise, anomalies,  distributional shifts, and adversarial attacks.

## Simple installing

python -m venv env

.\env\Scripts\activate

pip install -r requirements.txt

## Usage

python main.py threat_type 1 0 0 path_for_saving_data
