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
from ferramentas.Gerador_series import Gerador_conceitos


######################## Example of how to generate and plot the graph #########################################################################################

# 1. Generation parameters ###################
tamanho_conceitos = 2000
qtd_series = 20
grafico = True
# 1. #########################################
    

# 2. Importing the generator #################
gerador = Gerador_conceitos()
# 2. #########################################
        

# 3. Time series generator ########################################################################################

# Linear gradual time series
[pasta, nome, serie] = gerador.series_lineares_graduais_revista(tamanho_conceitos, qtd_series, grafico)

# Linear abrupt time series
#[pasta, nome, serie] = gerador.series_lineares_abruptas_revista(tamanho_conceitos, qtd_series, grafico)

# Non-linear gradual time series
#[pasta, nome, serie] = gerador.series_nlineares_graduais_revista(tamanho_conceitos, qtd_series, grafico)

# Non-linear abrupt time series
#[pasta, nome, serie] = gerador.series_nlineares_abruptas_revista(tamanho_conceitos, qtd_series, grafico)

# Sazonal time series
#[pasta, nome, serie] = gerador.series_sazonais_ictai(tamanho_conceitos, qtd_series, grafico)

# Hibrid time series
#[pasta, nome, serie] = gerador.series_hibridas_ictai(tamanho_conceitos, qtd_series, grafico)

# Linear time series
#[pasta, nome, serie] = gerador.series_lineares_ictai(tamanho_conceitos, qtd_series, grafico)

# Non-Linear time series
#[pasta, nome, serie] = gerador.series_nlineares_ictai(tamanho_conceitos, qtd_series, grafico)

# 3. ##############################################################################################################

# 4. Method for storing time series #######################################
# OBS: The generated series are stored in the "series/series geradas" directory
gerador.Escrever_serie_csv(pasta, nome+"_number", serie)  
# 4. ######################################################################
