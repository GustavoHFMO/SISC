'''
Created on 6 de fev de 2017

By Gustavo Oliveira
Universidade Federal de Alagoas, Penedo, Brasil
E-mail: gustavo.oliveira@penedo.ufal.br

ALGORITHMS USED IN THE MASTER DISSERTATION PUBLISHED BELOW:

OLIVEIRA, Gustavo Henrique Ferreira de Miranda. 
Previsão de séries temporais na presença de mudança de conceito: uma abordagem baseada em PSO. 
2018. Dissertação de Mestrado. Universidade Federal de Pernambuco.
https://repositorio.ufpe.br/handle/123456789/29987

SISC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
'''

# importing the algorithms
from algoritmos_online.ELM_DDM import ELM_DDM
from algoritmos_online.ELM_ECDD import ELM_ECDD
from algoritmos_online.ELM_FEDD import ELM_FEDD
from algoritmos_online.IDPSO_ELM_B import IDPSO_ELM_B
from algoritmos_online.IDPSO_ELM_S import IDPSO_ELM_S
from algoritmos_online.SISC_M import SISC_M
from algoritmos_online.SISC_P import SISC_P
from algoritmos_online.RPSO_ELM import RPSO_ELM
from algoritmos_online.ELM_SD import ELM_SD

# importing libs to help in the methods execution
from ferramentas.Importar_dataset import Datasets
from ferramentas.Particionar_series import Particionar_series

#1. importing the datasets
dtst = Datasets()


#1.1 REAL DATASETS ###################################################
# 1 = Dow 30
# 2 = Nasdaq
# 3 = S&P 500
dataset = dtst.Leitura_dados(dtst.bases_reais(1), csv=True)
dataset = Particionar_series().Normalizar(dataset)
#1.1 #################################################################

#1.2 REAL DATASETS WITH KNOWN DRIFTS #################################
# 1 = Dow-drift
# 2 = S&P500-drift
#dataset = dtst.Leitura_dados(dtst.bases_reais_drift(2), csv=True, column = 1)
#dataset = Particionar_series().Normalizar(dataset)
#1.2 #################################################################

#1.3 SYNTHETIC DATASETS ###############################################
# dtst.bases_linear_abruptas(1)
# dtst.bases_linear_graduais(1)
# dtst.bases_nlinear_abruptas(1)
# dtst.bases_nlinear_graduais(1)
# dtst.bases_sazonais(1)
#dataset = dtst.Leitura_dados(dtst.bases_hibridas(1), csv=True)
#dataset = Particionar_series().Normalizar(dataset)
#1.3 ##################################################################


#2. IMPORTING AND RUNNING ALGORITHMS ################################
# SISC-P
sisc_p = SISC_P(dataset)
sisc_p.Executar(grafico=True)

# SISC-M
sisc_m = SISC_M(dataset)
sisc_m.Executar(grafico=True)

# RPSO-ELM
rpso_elm = RPSO_ELM(dataset)
rpso_elm.Executar(grafico=True)

# IDPSO-ELM-S
idpso_elm_s = IDPSO_ELM_S(dataset)
idpso_elm_s.Executar(grafico=True)

# IDPSO-ELM-B
idpso_elm_b = IDPSO_ELM_B(dataset)
idpso_elm_b.Executar(grafico=True)

# ELM-FEDD
elm_fedd = ELM_FEDD(dataset)
elm_fedd.Executar(grafico=True)

# ELM-ECDD
elm_ecdd = ELM_ECDD(dataset)
elm_ecdd.Executar(grafico=True)

# ELM-DDM
elm_ddm = ELM_DDM(dataset)
elm_ddm.Executar(grafico=True)

# ELM
elm_sd = ELM_SD(dataset)
elm_sd.Executar(grafico=True)
#2. ###################################################################

####### 3. STORING THE PREDICTIONS ################################################################
import pandas as pd
df = pd.DataFrame(data={'predictions': idpso_elm_s.predictions, 'target':idpso_elm_s.target})
df.to_csv("images/"+idpso_elm_s.tecnica+".csv")
####### 3. STORING THE PREDICTIONS ################################################################













    
