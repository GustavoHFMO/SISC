#-*- coding: utf-8 -*-

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


# importing the tools 
import matplotlib.pyplot as plt
from ferramentas.Particionar_series import Particionar_series
from regressores.ELM import ELMRegressor
from regressores.IDPSO_ELM import IDPSO_ELM
from regressores.PSO_SLFN import PSO_SLFN
from regressores.PSO_ELM import PSO_ELM
from sklearn.metrics import mean_absolute_error
from ferramentas.Importar_dataset import Datasets


# 1. Importing the dataset ################################################################
dtst = Datasets('fora')
serie = dtst.Leitura_dados(dtst.bases_nlinear_abruptas(2), csv=True)
particao = Particionar_series(serie, [0.0, 0.0, 0.0], 0)
serie = particao.Normalizar(serie)
serie = serie[0:1000]
# 1. ######################################################################################

# 2. Dataset division settings and model parameters #######################################
divisao_dataset = [0.6, 0.2, 0.2]
qtd_neuronis = 5
janela_tempo = 5
inercia_inicial = 0.8
inercia_final = 0.2
# 2. ######################################################################################


# 3. Training the regressors #########################################################

# IDPSO+ELM
idpso_elm = IDPSO_ELM(serie, divisao_dataset, janela_tempo, qtd_neuronis)
idpso_elm.Parametros_IDPSO(100, 30, inercia_inicial, inercia_final, 2, 2, 1, 20)
idpso_elm.Treinar()  

# ELM
elm = ELMRegressor(qtd_neuronis)
elm.Tratamento_dados(serie, divisao_dataset, janela_tempo)
elm.Treinar(elm.train_entradas, elm.train_saidas)

# PSO+MLP
pso_slfn = PSO_SLFN(serie, divisao_dataset, janela_tempo, 3)
pso_slfn.Parametros_PSO(100, 30, inercia_inicial, 2, 2, 20, 0.25)
pso_slfn.Treinar()

# PSO+ELM
pso_elm = PSO_ELM(serie, divisao_dataset, janela_tempo, qtd_neuronis)
pso_elm.Parametros_PSO(100, 30, inercia_inicial, inercia_inicial, 2, 2, 1, 20, 0.25)
pso_elm.Treinar()

# 3. ######################################################################################

# organizando os dados para comparacao #
train_x, train_y = idpso_elm.dataset[0], idpso_elm.dataset[1] 
val_x, val_y = idpso_elm.dataset[2], idpso_elm.dataset[3]
test_x, test_y = idpso_elm.dataset[4], idpso_elm.dataset[5]



# 4. Ploting training predictions #########################################################

plt.plot(train_y, label = "Real")

print("\n------------------------------------------")

previsao = idpso_elm.Predizer(train_x)
MAE = mean_absolute_error(train_y, previsao)
plt.plot(previsao, label = "Previsao IDPSO-ELM: " + str(MAE))
print('IDPSO_ELM - Train MAE: ', MAE)


previsao = elm.Predizer(train_x)
MAE = mean_absolute_error(train_y, previsao)
plt.plot(previsao, label = "Previsao ELM: " + str(MAE))
print('ELM - Train MAE: ', MAE)

previsao = pso_slfn.Predizer(train_x)
MAE = mean_absolute_error(train_y, previsao)
plt.plot(previsao, label = "Previsao PSO_SLFN: " + str(MAE))
print('PSO_SLFN - Train MAE: ', MAE)


previsao = pso_elm.Predizer(train_x)
MAE = mean_absolute_error(train_y, previsao)
plt.plot(previsao, label = "Previsao PSO_ELM: " + str(MAE))
print('PSO_ELM - Train MAE: ', MAE)

print("------------------------------------------")

plt.legend()
plt.show()

# 4. ######################################################################################


# 5. Ploting validation predictions #######################################################

plt.plot(val_y, label = "Real")

print("\n------------------------------------------")

previsao = idpso_elm.Predizer(val_x)
MAE = mean_absolute_error(val_y, previsao)
plt.plot(previsao, label = "Previsao IDPSO-ELM: " + str(MAE))
print('IDPSO_ELM - val MAE: ', MAE)


previsao = elm.Predizer(val_x)
MAE = mean_absolute_error(val_y, previsao)
plt.plot(previsao, label = "Previsao ELM: " + str(MAE))
print('ELM - val MAE: ', MAE)


previsao = pso_slfn.Predizer(val_x)
MAE = mean_absolute_error(val_y, previsao)
plt.plot(previsao, label = "Previsao PSO_SLFN: " + str(MAE))
print('PSO_SLFN - val MAE: ', MAE)


previsao = pso_elm.Predizer(val_x)
MAE = mean_absolute_error(val_y, previsao)
plt.plot(previsao, label = "Previsao PSO_ELM: " + str(MAE))
print('PSO_ELM - val MAE: ', MAE)

print("------------------------------------------")

plt.legend()
plt.show()

# 5. ######################################################################################


# 6. Ploting test predictions #######################################################

plt.plot(test_y, label = "Real")

print("\n------------------------------------------")

previsao = idpso_elm.Predizer(test_x)
MAE = mean_absolute_error(test_y, previsao)
plt.plot(previsao, label = "Previsao IDPSO-ELM: " + str(MAE))
print('IDPSO_ELM - test MAE: ', MAE)


previsao = elm.Predizer(test_x)
MAE = mean_absolute_error(test_y, previsao)
plt.plot(previsao, label = "Previsao ELM: " + str(MAE))
print('ELM - test MAE: ', MAE)


previsao = pso_slfn.Predizer(test_x)
MAE = mean_absolute_error(test_y, previsao)
plt.plot(previsao, label = "Previsao PSO_SLFN: " + str(MAE))
print('PSO_SLFN - test MAE: ', MAE)


previsao = pso_elm.Predizer(test_x)
MAE = mean_absolute_error(test_y, previsao)
plt.plot(previsao, label = "Previsao PSO_ELM: " + str(MAE))
print('PSO_ELM - test MAE: ', MAE)

print("------------------------------------------")

plt.legend()
plt.show()

# 6. ######################################################################################



