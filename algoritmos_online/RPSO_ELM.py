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

from ferramentas.Janela_deslizante import Janela
from ferramentas.Importar_dataset import Datasets
from ferramentas.Particionar_series import Particionar_series
from metricas.Metricas_deteccao import Metricas_deteccao
from graficos.Graficos_execucao import Grafico
from sklearn.metrics import mean_absolute_error
from regressores.PSO_ELM import PSO_ELM
import time
import numpy as np


#parametros IDPSO
it = 50
inercia = 0.8
xmax = 1
c1 = 2
c2 = 2
crit_parada = 2
divisao_dataset = [0.8, 0.2, 0]

class RPSO_ELM():
    def __init__(self, dataset, m=100, lags=5, qtd_neuronios=5, numero_particulas=30, S=1, tx=0.25, w=1):
        '''
        construtor do algoritmo que detecta a mudanca de ambiente por meio de sensores
        :param dataset: serie temporal que o algoritmo vai executar
        :param qtd_train_inicial: quantidade de exemplos para o treinamento inicial
        :param tamanho_janela: tamanho da janela de caracteristicas para identificar a mudanca
        :param n: tamanho do n para reavaliar o metodo de deteccao
        :param lags: quantidade de lags para modelar as entradas da RNA
        :param qtd_neuronios: quantidade de neuronios escondidos da RNA
        :param numero_particulas: numero de particulas para serem usadas no IDPSO
        :param qtd_sensores: quantidade de sensores utilizados para detectar uma mudanca de conceito
        :param S: quantidade de passos para avaliar uma nova janela de solucoes
        :param c: c para medir se houve uma mudanca de conceito
        :param tx: porcentagem de particulas que serão reinicializadas
        '''
        
        self.dataset = dataset
        self.m = m
        self.lags = lags
        self.qtd_neuronios = qtd_neuronios
        self.numero_particulas = numero_particulas
        
        self.S = S
        self.tx = tx
        self.w = w
       
    def Sentry(self, solucao):
        '''
        metodo para salvar a solucao de uma particula sentry 
        :param: solucao: fitnesse de um primeiro conceito
        '''
        
        self.solucao = solucao
        
    def Monitorar_sentry(self, nova_solucao):
        '''
        metodo para monitorar a mudanca de conceito 
        :param: nova_solucao: novo fitness que sera comparado com um anterior
        :return: retorna verdadeiro caso uma mudança de conceito tenha ocorrido
        '''
       
        if(nova_solucao > (self.solucao[0] + (self.solucao[1]*self.w))):
            return True
        else:
            return False
         
    def Computar_solucao_teste(self, dados, lags, enxame): 
        '''
        Este método tem por objetivo computar o fitness de uma particula sentry para um determinado conceito 
        :param dados: dados que serao usados para definir um conceito
        :param lags: quantidade de lags para modelar a entrada da redes
        :param enxame: enxame que possui a particula sentry 
        :return: retorna o fitness para o conjunto de dados passados  
        '''    
        
        # modelando os dados e realizando as previsoes
        particao = Particionar_series(dados, [1, 0, 0], lags)
        [caracteristicas_entrada, caracteristicas_saida] = particao.Part_train()
        previsoes = enxame.Predizer(Entradas=caracteristicas_entrada)
        
        #calculando a estatistica do sensor
        acuracias = []
        for i in range(len(caracteristicas_entrada)):
            erro = mean_absolute_error(caracteristicas_saida[i:i+1], previsoes[i:i+1])
            acuracias.append(erro)
            
        estatistica = [0] * 2
        estatistica[0] = np.mean(acuracias)
        estatistica[1] = np.std(acuracias)
        
        return estatistica
    
    def Reavaliar_sentry(self, dados, lags, enxame): 
        '''
        Este método tem por objetivo computar o fitness de uma particula sentry para um determinado conceito 
        :param dados: dados que serao usados para definir um conceito
        :param lags: quantidade de lags para modelar a entrada da redes
        :param enxame: enxame que possui a particula sentry 
        :return: retorna o fitness para o conjunto de dados passados  
        '''    
        
        # modelando os dados e realizando as previsoes
        particao = Particionar_series(dados, [1, 0, 0], lags)
        [caracteristicas_entrada, caracteristicas_saida] = particao.Part_train()
        previsoes = enxame.Predizer(Entradas=caracteristicas_entrada)
        
        return mean_absolute_error(caracteristicas_saida, previsoes)
    
    def Executar(self, grafico = None):
        '''
        Metodo para executar o procedimento do algoritmo
        :param grafico: variavel booleana para ativar ou desativar o grafico
        :return: retorna 5 variaveis: [falsos_alarmes, atrasos, falta_deteccao, MAPE, tempo_execucao]
        '''
        
        ################################################################################################################################################
        ################################# CONFIGURACAO DO DATASET ######################################################################################
        ################################################################################################################################################
        
        #dividindo os dados da dataset dinamica para treinamento_inicial inicial e para uso do stream dinâmico
        treinamento_inicial = self.dataset[0:self.m]
        stream = self.dataset[self.m:]
    
        ################################################################################################################################################
        ################################# PERIODO ESTATICO #############################################################################################
        ################################################################################################################################################
        
        #criando e treinando um modelo_vigente para realizar as previsões
        enxame = PSO_ELM(treinamento_inicial, divisao_dataset, self.lags, self.qtd_neuronios)
        enxame.Parametros_PSO(it, self.numero_particulas, inercia, inercia, c1, c2, xmax, crit_parada, self.tx)
        enxame.Treinar()  
       
        #ajustando com os dados finais do treinamento a janela de predicao
        janela_predicao = Janela()
        janela_predicao.Ajustar(enxame.dataset[0][(len(enxame.dataset[0])-1):])
        predicao = enxame.Predizer(janela_predicao.dados)
        
        #janela com o atual conceito, tambem utilizada para armazenar os dados de retreinamento
        janela_caracteristicas = Janela()
        janela_caracteristicas.Ajustar(treinamento_inicial)
    
        #ativando os sensores de acordo com a primeira janela de caracteristicas
        solucao = self.Computar_solucao_teste(janela_caracteristicas.dados, self.lags, enxame)
        self.Sentry(solucao)
        ################################################################################################################################################
        ################################# PERIODO DINAMICO #############################################################################################
        ################################################################################################################################################
        
        #variavel para armazenar o erro do stream
        erro_stream = 0
        #variavel para armazenar as deteccoes
        deteccoes = []
        #variavel para armazenar os alarmes
        alarmes = []
        #variavel para armazenar o tempo inicial
        start_time = time.time()
        
        #vetor para armazenar a predicoes_vetor
        if(grafico == True):
            predicoes_vetor = [None] * len(stream)
            erro_stream_vetor = [None] * len(stream)
            
        #variavel auxiliar 
        mudanca_ocorreu = False
        
        #entrando no stream de dados
        for i in range(1, len(stream)):
            
            #computando o erro
            loss = mean_absolute_error(stream[i:i+1], predicao)
            erro_stream += loss
    
            #adicionando o novo dado a janela de predicao
            janela_predicao.Add_janela(stream[i])
                
            #realizando a nova predicao com a nova janela de predicao
            predicao = enxame.Predizer(janela_predicao.dados)
            
            # atualizando a janela de caracteristicas
            janela_caracteristicas.Add_janela(stream[i])
            
            
            if(grafico == True):                
                #salvando o erro 
                erro_stream_vetor[i] = loss
                #salvando a predicao
                predicoes_vetor[i] = predicao[0]
                
            #print("[", i, "]")
            
            if(mudanca_ocorreu == False):
                
                if((i % self.S) == 0):
                    #verificando os sensores
                    nova_solucao = self.Reavaliar_sentry(janela_caracteristicas.dados[0], self.lags, enxame)
                    mudou = self.Monitorar_sentry(nova_solucao)

                    if(mudou):
                        if(grafico == True):    
                            print("[%d] Mudança" % (i))
                        deteccoes.append(i)
                        
                        #variavel para alterar o fluxo, ir para o periodo de retreinamento
                        mudanca_ocorreu = True
                    
            else:
                
                #atualizando o modelo_vigente preditivo
                dataset = enxame.Tratamento_Dados(janela_caracteristicas.dados[0], divisao_dataset, self.lags)
                enxame.dataset = dataset
                enxame.Retreinar() 
                    
                #ajustando com os dados finais do treinamento a janela de predicao
                janela_predicao = Janela()
                janela_predicao.Ajustar(enxame.dataset[0][(len(enxame.dataset[0])-1):])
                predicao = enxame.Predizer(janela_predicao.dados)
                    
                #ativando os sensores de acordo com a primeira janela de caracteristicas
                solucao = self.Computar_solucao_teste(janela_caracteristicas.dados[0], self.lags, enxame)
                self.Sentry(solucao)
                    
                #variavel para voltar para o loop principal
                mudanca_ocorreu = False
        
        #variavel para armazenar o tempo final
        end_time = time.time()
        
        #computando as metricas de deteccao
        mt = Metricas_deteccao()
        [falsos_alarmes, atrasos] = mt.resultados(stream, deteccoes, self.m)
        
        #computando a acuracia da previsao ao longo do fluxo de dados
        MAE = erro_stream/len(stream)
        
        #computando o tempo de execucao
        tempo_execucao = (end_time-start_time)
        
        if(grafico == True):
            tecnica = "RPSO-ELM"
            print(tecnica)
            print("Alarmes:")
            print(alarmes)
            print("Deteccoes:")
            print(deteccoes)
            print("Falsos Alarmes: ", falsos_alarmes)
            print("Atrasos: ", atrasos)
            print("MAE: ", MAE)
            print("Tempo de execucao: ", tempo_execucao)
        
        #plotando o grafico de erro
        if(grafico == True):
            g = Grafico()
            g.Plotar_graficos(stream, predicoes_vetor, deteccoes, alarmes, erro_stream_vetor, self.m, atrasos, falsos_alarmes, tempo_execucao, MAE, nome=tecnica)
                           
        #retorno do metodo
        return falsos_alarmes, atrasos, MAE, tempo_execucao
    
def main():
    
    #instanciando o dataset
    dtst = Datasets('fora')
    dataset = dtst.Leitura_dados(dtst.bases_reais(3), csv=True)
    particao = Particionar_series(dataset, [0.0, 0.0, 0.0], 0)
    dataset = particao.Normalizar(dataset)
        
    #instanciando o algoritmo com sensores
    M = 100
    lags = 5
    qtd_neuronios = 5
    numero_particulas = 30
    S = 1
    tx = 0.25
    w = 1
    
    alg = RPSO_ELM(dataset, M, lags, qtd_neuronios, numero_particulas, S, tx, w)
    alg.Executar(grafico=True)
    
    
if __name__ == "__main__":
    main()      