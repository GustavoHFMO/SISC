#-*- coding: utf-8 -*-
'''
Created on 29 de fev de 2017

@author: gusta
'''

from ferramentas.Janela_deslizante import Janela
from ferramentas.Importar_dataset import Datasets
from ferramentas.Particionar_series import Particionar_series
from metricas.Metricas_deteccao import Metricas_deteccao
from regressores.ELM import ELMRegressor
from graficos.Graficos_execucao import Grafico
from sklearn.metrics import mean_absolute_error
import time

divisao_dataset = [0.8, 0.2, 0]

class ELM_SD():
    def __init__(self, dataset, n=100, lags=5, qtd_neuronios=10):
        '''
        construtor do algoritmo que detecta a mudanca de ambiente por meio do comportamento das particulas
        :param dataset: serie temporal que o algoritmo vai executar
        :param qtd_train_inicial: quantidade de exemplos para o treinamento inicial
        :param tamanho_janela: tamanho da janela de caracteristicas para identificar a mudanca
        :param n: tamanho do n para reavaliar o metodo de deteccao
        :param lags: quantidade de lags para modelar as entradas da RNA
        :param qtd_neuronios: quantidade de neuronios escondidos da RNA
        :param limite: contador para verificar a mudanca
        '''
        
        self.dataset = dataset
        self.n = n
        self.lags = lags
        self.qtd_neuronios = qtd_neuronios
        
    def Executar(self, grafico = None):
        '''
        Metodo para executar o procedimento do algoritmo
        :param grafico: variavel booleana para ativar ou desativar o grafico
        :return: retorna 5 variaveis: [falsos_alarmes, atrasos, falta_deteccao, MAPE, tempo_execucao]
        '''

        ################################################################################################################################################
        ################################# CONFIGURACAO DO DATASET ######################################################################################
        ################################################################################################################################################
        
        #dividindo os dados da dataset dinamica para treinamento_inicial inicial e para uso do stream din�mico
        treinamento_inicial = self.dataset[0:self.n]
        stream = self.dataset[self.n:]
    
        ################################################################################################################################################
        ################################# PERIODO ESTATICO #############################################################################################
        ################################################################################################################################################
        
        #criando e treinando um enxame_vigente para realizar as previsoes
        ELM = ELMRegressor(self.qtd_neuronios)
        ELM.Tratamento_dados(treinamento_inicial, divisao_dataset, self.lags)
        ELM.Treinar(ELM.train_entradas, ELM.train_saidas)
        
        #ajustando com os dados finais do treinamento a janela de predicao
        janela_predicao = Janela()
        janela_predicao.Ajustar(ELM.train_entradas[len(ELM.train_entradas)-1:])
        predicao = ELM.Predizer(janela_predicao.dados)
        
        #janela com o atual conceito, tambem utilizada para armazenar os dados de retreinamento
        janela_caracteristicas = Janela()
        janela_caracteristicas.Ajustar(treinamento_inicial)
        
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
        
        #entrando no stream de dados
        for i in range(1, len(stream)):
            
            #computando o erro
            loss = mean_absolute_error(stream[i:i+1], predicao)
            erro_stream += loss
    
            #adicionando o novo dado a janela de predicao
            janela_predicao.Add_janela(stream[i])
                
            #realizando a nova predicao com a nova janela de predicao
            predicao = ELM.Predizer(janela_predicao.dados)
            
            #
            if(grafico == True):                
                #salvando o erro 
                erro_stream_vetor[i] = loss
                #salvando a predicao
                predicoes_vetor[i] = predicao[0]

                            
        #variavel para armazenar o tempo final
        end_time = time.time()
        
        #computando as metricas de deteccao
        mt = Metricas_deteccao()
        deteccoes.append(-500)
        [falsos_alarmes, atrasos] = mt.resultados(stream, deteccoes, self.n)
        
        #computando a acuracia da previsao ao longo do fluxo de dados
        MAE = erro_stream/len(stream)
        
        #computando o tempo de execucao
        tempo_execucao = (end_time-start_time)
        
        if(grafico == True):
            tecnica = "ELM"
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
            deteccoes.append(-500)
            g.Plotar_graficos(stream, predicoes_vetor, deteccoes, alarmes, erro_stream_vetor, self.n, atrasos, falsos_alarmes, tempo_execucao, MAE, nome=tecnica)
                           
        #retorno do metodo
        return falsos_alarmes, atrasos, MAE, tempo_execucao
    
def main():
    
    #instanciando o dataset
    dtst = Datasets('fora')
    dataset = dtst.Leitura_dados(dtst.bases_reais_drift(2), csv=True, column=1)
    particao = Particionar_series(dataset, [0.0, 0.0, 0.0], 0)
    dataset = particao.Normalizar(dataset)
                
    #instanciando o algoritmo com sensores
    n = 100
    lags = 5
    qtd_neuronios = 10 
    alg = ELM_SD(dataset, n, lags, qtd_neuronios)
    
    #colhendo os resultados
    for i in range(1):
        [falsos_alarmes, atrasos, MAE, tempo_execucao] = alg.Executar(grafico=True)
        print(MAE)
    
    
if __name__ == "__main__":
    main()      