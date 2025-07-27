#  Dynamic Swarm Intelligence for Time Series Forecasting in the Presence of Concept Drift
# [DOI](https://doi.org/10.1007/s42979-025-04247-z)

## Usage
```
# Cloning the repository
git clone https://github.com/GustavoHFMO/SISC.git

# Acessing the repository
cd SISC

# Installing the dependencies
pip install -r requirements.txt

# Running the codes (Description bellow)
python generate_synthetic_series.py
python run_regressors.py
python run_online_algorithms.py
```

1. The module [generate_synthetic_series.py](https://github.com/GustavoHFMO/SISC/blob/main/generate_synthetic_series.py) explains how to generate the synthetic series used in the work. Result:

![](https://github.com/GustavoHFMO/SISC/blob/main/images/time_series_generation.png)


2. The module [run_regressors.py](https://github.com/GustavoHFMO/SISC/blob/main/run_regressors.py) shows how to run the following regressors trained by swarm algorithms: `IDPSO+ELM`, `PSO+ELM` and `PSO+MLP`. Result:

![](https://github.com/GustavoHFMO/SISC/blob/main/images/Regressors_prediction.png)


3. The module [run_online_algorithms.py](https://github.com/GustavoHFMO/SISC/blob/main/run_online_algorithms.py) explain how to executes the algorithms described below in real and synthetic time series.

## [SISC-P](https://github.com/GustavoHFMO/SISC/blob/main/algoritmos_online/SISC_P.py)
> Proposed in the master dissertation.

## [SISC-M](https://github.com/GustavoHFMO/SISC/blob/main/algoritmos_online/SISC_M.py)
> Proposed in the master dissertation.

## [IDPSO-ELM-S:](https://github.com/GustavoHFMO/SISC/blob/master/algoritmos_online/IDPSO_ELM_S.py)
> OLIVEIRA, Gustavo HFMO et al. Time series forecasting in the presence of concept drift: A pso-based approach. In: 2017 IEEE 29th International Conference on Tools with Artificial Intelligence (ICTAI). IEEE, 2017. p. 239-246.

## [IDPSO-ELM-B:](https://github.com/GustavoHFMO/SISC/blob/master/algoritmos_online/IDPSO_ELM_B.py)
> OLIVEIRA, Gustavo HFMO et al. Time series forecasting in the presence of concept drift: A pso-based approach. In: 2017 IEEE 29th International Conference on Tools with Artificial Intelligence (ICTAI). IEEE, 2017. p. 239-246.

## [ELM-FEDD:](https://github.com/GustavoHFMO/SISC/blob/master/algoritmos_online/ELM_FEDD.py)
> R. C. Cavalcante, L. L. Minku, and A. L. Oliveira, “FEDD: Feature Extraction for Explicit Concept Drift Detection in time series,” in Neural Networks (IJCNN), 2016 International Joint Conference on. IEEE, 2016, pp. 740–747.

## [ELM-ECDD:](https://github.com/GustavoHFMO/SISC/blob/master/algoritmos_online/ELM_ECDD.py)
> R. C. Cavalcante and A. L. Oliveira, “An approach to handle concept drift in financial time series based on extreme learning machines and explicit drift detection,” in Neural Networks (IJCNN), 2015 International Joint Conference on. IEEE, 2015, pp. 1–8.

## [ELM-DDM:](https://github.com/GustavoHFMO/SISC/blob/master/algoritmos_online/ELM_DDM.py)
> R. C. Cavalcante and A. L. Oliveira, “An approach to handle concept drift in financial time series based on extreme learning machines and explicit drift detection,” in Neural Networks (IJCNN), 2015 International Joint Conference on. IEEE, 2015, pp. 1–8.

## [RPSO-ELM:](https://github.com/GustavoHFMO/SISC/blob/main/algoritmos_online/RPSO_ELM.py)
> RAKITIANSKAIA, A. S.; ENGELBRECHT, A. P. Training feedforward neural networks with dynamic particle swarm optimisation. Swarm Intelligence, Springer, v. 6, n. 3, p. 233–270, 2012. 

![](https://github.com/GustavoHFMO/SISC/blob/main/images/SISC_P_execution.png)

## License
This project is under a GNU General Public License (GPL) Version 3. See [LICENSE](https://www.gnu.org/licenses/gpl-3.0-standalone.html) for more information.
