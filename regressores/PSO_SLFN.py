#-*- coding: utf-8 -*-

'''
Created on 25 de set de 2017

@author: gusta
'''

import random
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from numpy import array
from ferramentas.Particionar_series import Particionar_series
from sklearn.metrics import mean_squared_error
from ferramentas.Importar_dataset import Datasets

#limites
Xmax = 5
Xmin = -Xmax
posMax = 1
posMin = -posMax

#variaveis auxiliares
contador = 0
fitness = 0
grafico = []

class SLFN():
    def __init__(self, NI, NH, pesos):
        '''
        classe para criar uma rede neural do uma camada feed foward
        :param: NI: lags de entrada, quantidade de neuronios para a camadada de entradas.
        :param: NH: quantidade de neuronios para a camada escondida.
        :param: pesos: conjunto de pesos que será atribuido por PSO.
        '''
        
        # número de camadas escondidas
        self.neur_entradas = NI
        self.neur_escondidos = NH
        self.neur_saida = 1
        
        # inicializando os bias os nós de ativação
        self.ativacao_inicial = [1.0] * self.neur_entradas
        self.ativacao_escondida = [1.0] * self.neur_escondidos
        self.ativacao_saida = [1.0] * self.neur_saida
    
        # dividindo os pesos para inicio e fim da rede
        self.pesos_iniciais = pesos[0:NI*NH]
        self.pesos_saida = pesos[NI*NH:]
    
        # criando as matrizes de pesos dos nos
        self.pesos_iniciais = self.pesos_iniciais.reshape(self.neur_entradas, self.neur_escondidos)
        self.pesos_saida = self.pesos_saida.reshape(self.neur_escondidos, self.neur_saida)
        
    def Executar_NN (self, dados_entradas):
        '''
        Metodo para computar os pesos da rede em um conjunto de entrada
        :param: dados_entradas: dados de entrada para ser computado
        :return: [erro para o conjunto de entrada, previsao]
        '''
        
        #atribui todos os dados_entradas os seus valores, somente o primeiro fica com o valor do bias
        for i in range(self.neur_entradas):
            self.ativacao_inicial[i] = dados_entradas[i]
      
        #multiplica as dados_entradas pelos seus pesos correspondentes, salva neur_escondidos no da camada escondida (ativacao_escondida) e ativa com a funcao sigmoide
        for j in range(self.neur_escondidos):
            sum = 0.0
            for i in range(self.neur_entradas):
                sum += (self.ativacao_inicial[i] * self.pesos_iniciais[i][j])
            self.ativacao_escondida[j] = self.Sigmoide(sum)
    
        #multiplica os dados da camada escondida para a saida
        for k in range(self.neur_saida):
            sum = 0.0
            for j in range(self.neur_escondidos):        
                sum += (self.ativacao_escondida[j] * self.pesos_saida[j][k])
            self.ativacao_saida[k] = sum
        
        #retorna a saida      
        return self.ativacao_saida[0]
    
    def Sigmoide(self, x):
        '''
        metodo para computar a funcao sigmoide para um determinado valor
        :param x: valor que será computado
        :return: valor computado
        '''
        return 1 / (1 + math.exp(-x))
    
    def Criar_matriz(self, I, J):
        '''
        metodo para criar uma matriz com numeros
        :param I: numero de linhas
        :param J: numero de colunas
        :return: retorna uma matriz de zeros
        '''
        
        m = []
        for i in range(I):
            m.append([0.0] * J)
        return m

    #método para dar valores aleatorios a matriz de zeros, a e b são os intervalos

class Particulas():
    '''
    classe para criar as particulas
    '''
    pass

class PSO_SLFN():
    def __init__(self, serie, divisao, janela, qtd_neuronios):
        '''
        Contrutor para o algoritmo de treinamento do ELM, o algoritmo utilizado e o IDPSO
        :param serie: vetor, com a serie temporal utilizada para treinamento 
        :param divisao: lista com porcentagens, da seguinte forma [pct_treinamento_entrada, pct_treinamento_saida, pct_validacao_entrada, pct_validacao_saida]
        :param janela: quantidade de lags usados para modelar os padroes de entrada da ELM
        :param qtd_neuronios: quantidade de neuronios da camada escondida da ELM
        '''
        
        #serie = vetor
        #divisao = lista com três porcentagens para divisao da serie
        #janela = quantidade de lags
        #qtd_neuronios = quantidade de neuronios
        
        #tratando os dados
        #dataset = [treinamento_entrada, treinamento_saida, validacao_entrada, valic_saida, teste_entrada, teste_saida]
        dataset = self.Tratamento_Dados(serie, divisao, janela)
        
        self.dataset = dataset
        self.qtd_neuronios = qtd_neuronios
        
        #default PSO
        self.linhas = self.dataset[0].shape[1]
        self.numero_dimensoes =  self.linhas * qtd_neuronios + qtd_neuronios
        
        self.iteracoes = 1000
        self.numero_particulas = 30
        self.inercia = 0.5
        self.c1 = 2.4
        self.c2 = 1.4
        self.crit_parada = 50
        self.particulas = []
        self.gbest = []
        
        self.tx_espalhar = 0.25
        
    def Parametros_PSO(self, iteracoes, numero_particulas, inercia, c1, c2, crit_parada, tx):
        '''
        Metodo para alterar os parametros basicos do IDPSO 
        :param iteracoes: quantidade de geracoes para o treinamento 
        :param numero_particulas: quantidade de particulas usadas para treinamento
        :param inercia: inercial inicial para treinamento
        :param inercia_final: inercia final para variacao
        :param c1: coeficiente cognitivo
        :param c2: coeficiente pessoal
        :param crit_parada: criterio de parada para limitar a repeticao nao melhora do gbest
        '''
        
        self.iteracoes = iteracoes
        self.numero_particulas = numero_particulas
        self.c1 = c1
        self.c2 = c2
        self.crit_parada = crit_parada
        self.tx_espalhar = tx
    
    def Tratamento_Dados(self, serie, divisao, janela):
        '''
        Metodo para dividir a serie temporal em treinamento e validacao 
        :param serie: vetor, com a serie temporal utilizada para treinamento 
        :param divisao: lista com porcentagens, da seguinte forma [pct_treinamento_entrada, pct_treinamento_saida, pct_validacao_entrada, pct_validacao_saida]
        :param janela: quantidade de lags usados para modelar os padroes de entrada da ELM
        :return: retorna uma lista com os seguintes dados [treinamento_entrada, treinamento_saida, validacao_entrada, validacao_saida]
        '''
        
        #tratamento dos dados
        particao = Particionar_series(serie, divisao, janela)
        [train_entrada, train_saida] = particao.Part_train()
        [val_entrada, val_saida] = particao.Part_val()
        [test_entrada, test_saida] = particao.Part_test()
        
        #inserindo os dados em uma lista
        lista_dados = []
        lista_dados.append(train_entrada)
        lista_dados.append(train_saida)
        lista_dados.append(val_entrada)
        lista_dados.append(val_saida)
        lista_dados.append(test_entrada)
        lista_dados.append(test_saida)
        
        #retornando o valor
        return lista_dados
      
    def Criar_Particula(self):
        '''
        Metodo para criar todas as particulas do enxame de forma aleatoria 
        '''
        
        for i in range(self.numero_particulas):
            p = Particulas()
            p.posicao = np.random.randn(1, self.numero_dimensoes)
            p.posicao = p.posicao[0]
            p.fitness = self.Funcao(p.posicao)
            p.velocidade = array([0.0 for i in range(self.numero_dimensoes)])
            p.best = p.posicao
            p.fit_best = p.fitness
            self.particulas.append(p)
        
        self.gbest = self.particulas[0]
        
    def Funcao(self, posicao):
        '''
        Metodo para calcular a funcao objetivo do IDPSO, nesse caso a funcao e a previsao de um ELM 
        :param posicao: posicao seria os pesos da camada de entrada e os bias da rede ELM 
        :return: retorna o MSE obtido da previsao de uma ELM
        '''
        
        # criando uma rede neural nova
        sl = SLFN(self.linhas, self.qtd_neuronios, posicao)
        
        # realizando a previsao para cada padrao de treinamento
        previsao = []
        for i in range(len(self.dataset[0])):
            predicao = sl.Executar_NN(self.dataset[0][i])
            previsao.append(predicao)
        
        # computando o erro de previsao para os dados de treinamento
        return mean_squared_error(self.dataset[1], previsao)
    
    def Fitness(self):
        '''
        Metodo para computar o fitness de todas as particulas 
        '''
        
        for i in self.particulas:   
            i.fitness = self.Funcao(i.posicao)
        
    def Velocidade(self):
        '''
        Metodo para computar a velocidade de todas as particulas 
        '''
        
        calculo_c1 = 0
        calculo_c2 = 0
        
        for i in self.particulas:
            for j in range(len(i.posicao)):
                calculo_c1 = (i.best[j] - i.posicao[j])
                calculo_c2 = (self.gbest.posicao[j] - i.posicao[j])
                
                influecia_inercia = (self.inercia * i.velocidade[j])
                influencia_cognitiva = ((self.c1 * random.random()) * calculo_c1)
                influecia_social = ((self.c2 * random.random()) * calculo_c2)
              
                i.velocidade[j] = influecia_inercia + influencia_cognitiva + influecia_social
                
                if (i.velocidade[j] >= Xmax):
                    i.velocidade[j] = Xmax
                elif(i.velocidade[j] <= Xmin):
                    i.velocidade[j] = Xmin
              
    def Atualizar_particulas(self):
        '''
        Metodo para atualizar a posicao de todas as particulas 
        '''
        
        for i in self.particulas:
            for j in range(len(i.posicao)):
                i.posicao[j] = i.posicao[j] + i.velocidade[j]
                
                if (i.posicao[j] >= posMax):
                    i.posicao[j] = posMax
                elif(i.posicao[j] <= posMin):
                    i.posicao[j] = posMin

    def Pbest(self):
        '''
        Metodo para computar os pbests das particulas  
        '''
        
        for i in self.particulas:
            if(i.fit_best >= i.fitness):
                i.best = i.posicao
                i.fit_best = i.fitness

    def Gbest(self):
        '''
        Metodo para computar o gbest do enxame  
        '''
        
        for i in self.particulas:
            if(i.fitness < self.gbest.fitness):
                self.gbest = copy.deepcopy(i)
    
    def Criterio_parada(self, erro, i):
        '''
        Metodo para analisar o erro do conjunto de validacao
        :param erro: erro atual do conjunto de validacao
        :return: retorna a indice da ultima geracao para parar o algoritmo  
        '''
        
        global contador, fitness
        
        # salvando o primeiro erro do conjunto de validacao
        if(i == 1):
            self.erro_val_min = copy.deepcopy(erro)
            fitness = copy.deepcopy(self.gbest.fitness)
            return i 
        
        # verificando a cada iteracao o erro do conjunto de validacao
        if(i > self.crit_parada):
            # computando o gl5
            gl5 = 100 * ((erro/self.erro_val_min) - 1)
            # conferinfo o erro em relacao ao gl5
            if(gl5 > 5):
                #print("[%d] GL5" % (i) + ": ", erro)
                return self.iteracoes
            
        else:
            # atualizando o erro de validacao caso ele seja menor 
            if(erro < self.erro_val_min):
                self.erro_val_min = copy.deepcopy(erro)
                
        # verificando se o fitness da melhor particula nao esta estacionario
        if(contador == self.crit_parada):
            return self.iteracoes
        else:
            if(fitness == self.gbest.fitness):
                contador+=1
                #print(contador)
            else:
                fitness = copy.deepcopy(self.gbest.fitness)
                contador = 0
                
        return i
        
    def Grafico_Convergencia(self, fitness, i):
        '''
        Metodo para apresentar o grafico de convergencia
        :param fitness: fitness da melhor particula da geracao
        :param i: atual geracao
        '''
        
        global grafico
        
        grafico.append(fitness)
        
        if(i == self.iteracoes):
            plt.plot(grafico)
            plt.title('Gráfico de Convergência')
            plt.show()
            
    def Predizer(self, Entradas, Saidas = None, grafico = None):
        '''
        Metodo para realizar a previsao com a melhor particula (ELM) do enxame e apresentar o grafico de previsao
        :param Entradas: padroes de entrada para realizar a previsao
        :param Saidas: padroes de saida para computar o MSE
        :param grafico: variavel booleana para ativar ou desativar o grafico de previsao
        :return: Retorna a predicao para as entradas apresentadas. Se as entradas e saidas sao apresentadas o MSE e retornado
        '''
        
        # computando a previsao
        sl = SLFN(self.linhas, self.qtd_neuronios, self.gbest.posicao)
        previsao = []
        for i in range(len(Entradas)):
            previsao.append(sl.Executar_NN(Entradas[i]))
                
        if(grafico == None):
            return previsao
        else:
            #retorna somente a previsao
            MSE = mean_squared_error(Saidas, previsao)
            print('MSE: %s' %MSE)
        
            #apresentar grafico
            if(grafico == True):
                plt.plot(Saidas, label = 'Real', color = 'Blue')
                plt.plot(previsao, label = 'Previsão', color = 'Red')
                plt.title('MSE: %s' %MSE)
                plt.legend()
                plt.tight_layout()
                plt.show()
                    
            return MSE
    
    def Treinar(self):
        '''
        Metodo para treinar a rede ELM com o IDPSO 
        '''
        
        #print("Treinando...")
        self.Criar_Particula()       
        
        i = 0
        while(i < self.iteracoes):
            i = i + 1
            
            self.Fitness()
            self.Gbest()
            self.Pbest()
            self.Velocidade()
            self.Atualizar_particulas()
            
            erro = self.Validacao()
            i = self.Criterio_parada(erro, i)
            
            '''
            print("[%d]" % (i) + ": ", self.gbest.fitness)
            self.Grafico_Convergencia(self.gbest.fitness, i)
            '''

    def Espalhar_particulas(self):
        '''
        método para apagar as informações de parte do enxame, essa parte é definida por uma porcentagem: self.tx_espalhar
        '''
        
        # zerando o contador do criterio de parada
        global contador 
        contador = 0
        
        # definindo a quantidade de particulas a serem apagadas
        qtd = len(self.particulas)
        tx = int(qtd * self.tx_espalhar) 
        escolhidos = []
        
        for i in range(tx):
            # gerando um numero aleatorio
            j = self.Gerar_numero(qtd-1, escolhidos)
            escolhidos.append(j)
            #print("Particulas reinicializadas: [", j, "]")
        
            # apagando a antiga particula
            del self.particulas[j]
            
            # criando uma nova e adicionando ela no enxame
            p = Particulas()
            p.posicao = np.random.randn(1, self.numero_dimensoes)
            p.posicao = p.posicao[0]
            p.fitness = self.Funcao(p.posicao)
            p.velocidade = array([0.0 for x in range(self.numero_dimensoes)])
            p.best = p.posicao
            p.fit_best = p.fitness
            self.particulas.append(p)
            
        # atualizando os pbests das particulas
        for i in self.particulas:
            i.fitness = self.Funcao(i.posicao)
            
        # atualizando o gbest
        self.gbest = self.particulas[0]
        
    def Gerar_numero(self, qtd, escolhidos):
        '''
        Método para gerar um numero aleatorio de forma que os valores não se repitam
        '''
        
        j = np.random.randint(0, qtd)
        
        if(j in escolhidos):
            
            return self.Gerar_numero(qtd, escolhidos)
            
        else:
            return j
    
    def Retreinar(self):
        '''
        Metodo para retreinar um modelo 
        '''
        
        #print("Retreinando...")
        self.Espalhar_particulas()
        
        i = 0
        while(i < self.iteracoes):
            i = i + 1
            
            self.Fitness()
            self.Gbest()
            self.Pbest()
            self.Velocidade()
            self.Atualizar_particulas()
            
            erro = self.Validacao()
            i = self.Criterio_parada(erro, i)
            
            '''
            print("[%d]" % (i) + " : ", self.gbest.fitness)
            self.Grafico_Convergencia(self.gbest.fitness, i)
            '''
            
    def Validacao(self):
        '''
        Metodo para calcular o erro do conjunto de validacao 
        :return: retorna o MSE obtido do conjunto de validacao
        '''
        
        # criando uma rede neural nova
        sl = SLFN(self.linhas, self.qtd_neuronios, self.gbest.posicao)
        
        # realizando a previsao para cada padrao de treinamento
        previsao = []
        for i in range(len(self.dataset[2])):
            predicao = sl.Executar_NN(self.dataset[2][i])
            previsao.append(predicao)
        
        # computando o erro de previsao para os dados de treinamento
        return mean_squared_error(self.dataset[3], previsao)
    
def main():
    #load da serie
    dtst = Datasets('dentro')
    serie = dtst.Leitura_dados(dtst.bases_reais(1), csv=True)
    particao = Particionar_series(serie, [0.0, 0.0, 0.0], 0)
    serie = particao.Normalizar(serie)
    serie = serie[0:100]

    '''
    serie1 = serie[10000:10300]
    serie2 = serie[12000:14000]
    
    modelo = IDPSO_ELM(serie1, [1, 0, 0], 5, 10)
    modelo.Parametros_IDPSO(100, 30, 0.8, 0.4, 2, 2, 20)
    modelo.Treinar()  
    
    previsao = modelo.Predizer(modelo.dataset[0])
    MAE = mean_absolute_error(modelo.dataset[1], previsao)
    print('IDPSO_ELM - MAE: ', MAE)
    
    plt.plot(serie1)
    plt.plot(previsao)
    plt.show()

    
    serie2_modelada = modelo.Tratamento_Dados(serie2, [1, 0, 0], 5)
    
    previsao = modelo.Predizer(serie2_modelada[0])
    MAE = mean_absolute_error(serie2_modelada[1], previsao)
    print('IDPSO_ELM - MAE: ', MAE)
    
    plt.plot(serie2)
    plt.plot(previsao)
    plt.show()
    '''
    
    divisao_dataset = [0.6, 0.2, 0.2]
    qtd_neuronis = 10
    janela_tempo = 5
    
    modelo = PSO_SLFN(serie, divisao_dataset, janela_tempo, qtd_neuronis)
    modelo.Parametros_PSO(100, 30, 0.8, 2, 2, 20, 0.25)
    modelo.Treinar() 
    
    #print(modelo.Predizer(Entradas=modelo.dataset[0]))
    modelo.Predizer(Entradas=modelo.dataset[0], Saidas=modelo.dataset[1], grafico=True)
    modelo.Predizer(Entradas=modelo.dataset[2], Saidas=modelo.dataset[3], grafico=True)
    modelo.Predizer(Entradas=modelo.dataset[4], Saidas=modelo.dataset[5], grafico=True)
    
    
if __name__ == "__main__":
    main()
    



