#-*- coding: utf-8 -*-
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from numpy import array
from ferramentas.Particionar_series import Particionar_series
from regressores.ELM import ELMRegressor
from sklearn.metrics import mean_absolute_error
from ferramentas.Importar_dataset import Datasets

#limites
mi = 100

#variaveis auxiliares
contador = 0
fitness = 0
grafico = []
lista_MSE = []

class Particulas():
    '''
    classe para criar as particulas
    '''
    pass

class PSO_ELM():
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
        self.best_elm = []
        
        #default IDPSO
        self.linhas = self.dataset[0].shape[1] + 1
        self.numero_dimensoes =  self.linhas * qtd_neuronios
        
        self.iteracoes = 1000
        self.numero_particulas = 30
        self.inercia = 0.5
        self.inercia_final = 0.3
        self.c1 = 2.4
        self.c2 = 1.4
        self.crit_parada = 50
        self.particulas = []
        self.gbest = []
        
        self.tx_espalhar = 0
        
    def Parametros_PSO(self, iteracoes, numero_particulas, inercia_inicial, inercia_final, c1, c2, Xmax, crit_parada, tx):
        '''
        Metodo para alterar os parametros basicos do IDPSO 
        :param iteracoes: quantidade de geracoes para o treinamento 
        :param numero_particulas: quantidade de particulas usadas para treinamento
        :param inercia: inercial inicial para treinamento
        :param c1: coeficiente cognitivo
        :param c2: coeficiente pessoal
        :param crit_parada: criterio de parada para limitar a repeticao nao melhora do gbest
        '''
        
        self.iteracoes = iteracoes
        self.numero_particulas = numero_particulas
        self.inercia_inicial = inercia_inicial
        self.inercia_final = inercia_final
        self.c1 = c1
        self.c2 = c2
        self.crit_parada = crit_parada
        self.tx_espalhar = tx
        
        self.xmax = Xmax
        self.xmin = -Xmax
        self.posMax = Xmax
        self.posMin = self.xmin
    
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
        
        global contador, fitness, grafico, lista_MSE
        contador = 0
        fitness = 0
        grafico = []
        lista_MSE = []
        
        for i in range(self.numero_particulas):
            p = Particulas()
            p.posicao = np.random.randn(1, self.numero_dimensoes)
            p.posicao = p.posicao[0]
            p.fitness = self.Funcao(p.posicao)
            p.velocidade = array([0.0 for i in range(self.numero_dimensoes)])
            p.best = p.posicao
            p.fit_best = p.fitness
            p.c1 = self.c1
            p.c2 = self.c2
            p.inercia = self.inercia
            p.phi = 0
            self.particulas.append(p)
        
        self.gbest = self.particulas[0]
        
    def Funcao(self, posicao):
        '''
        Metodo para calcular a funcao objetivo do IDPSO, nesse caso a funcao e a previsao de um ELM 
        :param posicao: posicao seria os pesos da camada de entrada e os bias da rede ELM 
        :return: retorna o MSE obtido da previsao de uma ELM
        '''
        
        # instanciando um modelo ELM
        ELM = ELMRegressor(self.qtd_neuronios)
        
        # modelando a dimensao das particulas para serem usadas 
        posicao = posicao.reshape(self.linhas, self.qtd_neuronios)
        
        # ELM treinando com a entrada e a saida do conjunto de treinamento e tambem com os pesos da particula 
        ELM.Treinar(self.dataset[0], self.dataset[1], posicao)
        
        # Realizando a previsao para o conjunto de validacao
        prediction_train = ELM.Predizer(self.dataset[2])
        
        # computando o erro do conjunto de validacao
        MAE_val = mean_absolute_error(self.dataset[3], prediction_train)
        
        # retornando o erro do conjunto de validacao - forma de evitar o overfitting
        return MAE_val
    
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
                
                influecia_inercia = (i.inercia * i.velocidade[j])
                influencia_cognitiva = ((i.c1 * random.random()) * calculo_c1)
                influecia_social = ((i.c2 * random.random()) * calculo_c2)
              
                i.velocidade[j] = influecia_inercia + influencia_cognitiva + influecia_social
                
                if (i.velocidade[j] >= self.xmax):
                    i.velocidade[j] = self.xmax
                elif(i.velocidade[j] <= self.xmin):
                    i.velocidade[j] = self.xmin
              
    def Atualizar_particulas(self):
        '''
        Metodo para atualizar a posicao de todas as particulas 
        '''
        
        for i in self.particulas:
            for j in range(len(i.posicao)):
                i.posicao[j] = i.posicao[j] + i.velocidade[j]
                
                if (i.posicao[j] >= self.posMax):
                    i.posicao[j] = self.posMax
                elif(i.posicao[j] <= self.posMin):
                    i.posicao[j] = self.posMin

    def Atualizar_parametros(self, iteracao):
        '''
        Metodo para atualizar os parametros: inercia, c1 e c2 
        '''
        
        for i in self.particulas:
            parte1 = 0
            parte2 = 0
            
            for j in range(len(i.posicao)):
                parte1 = parte1 + self.gbest.posicao[j] - i.posicao[j]
                parte2 = parte2 + i.best[j] - i.posicao[j]
                
                if(parte1 == 0):
                    parte1 = 1
                if(parte2 == 0):
                    parte2 = 1
                    
            i.phi = abs(parte1/parte2)
            
        for i in self.particulas:
            ln = np.log(i.phi)
            calculo = i.phi * (iteracao - ((1 + ln) * self.iteracoes) / mi)
            i.inercia = ((self.inercia - self.inercia_final) / (1 + np.exp(calculo))) + self.inercia_final
            i.c1 = self.c1 * (i.phi ** (-1))
            i.c2 = self.c2 * i.phi
            
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
            if(i.fitness <= self.gbest.fitness):
                self.gbest = copy.deepcopy(i)
    
    def Criterio_parada(self, i):
        '''
        Metodo para computar os criterios de parada, tanto o GL5 como o para nao melhora da melhor solucao
        :param i: atual geracao
        :return: retorna a indice da ultima geracao para parar o algoritmo  
        '''
        
        global contador, fitness, lista_MSE
        
        if(i == 0):
            fitness = copy.deepcopy(self.gbest.fitness)
            return i
        
        else:
            
            if(contador == self.crit_parada):
                #print("[%d] Sem melhora: " % (i) + " : ", self.gbest.fitness)
                return self.iteracoes
            
            if(fitness == self.gbest.fitness):
                contador+=1
                return i
            
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
            
    def Predizer(self, Entradas, num_sensor = None, Saidas = None, grafico = None):
        '''
        Metodo para realizar a previsao com a melhor particula (ELM) do enxame e apresentar o grafico de previsao
        :param Entradas: padroes de entrada para realizar a previsao
        :param Saidas: padroes de saida para computar o MSE
        :param grafico: variavel booleana para ativar ou desativar o grafico de previsao
        :return: Retorna a predicao para as entradas apresentadas. Se as entradas e saidas sao apresentadas o MSE e retornado
        '''
        
        # se o numero do sensor não é passado então a predição é feita com o gbest
        if(num_sensor == None):
        
            #retorna somente a previsao
            if(Saidas == None):
                prediction = self.best_elm.Predizer(Entradas)
                return prediction
            else:
                prediction = self.best_elm.Predizer(Entradas)
                MSE = mean_absolute_error(Saidas, prediction)
                print('\n MSE: %s' %MSE)
    
                #apresentar grafico
                if(grafico == True):
                    plt.plot(Saidas, label = 'Real', color = 'Blue')
                    plt.plot(prediction, label = 'Previsão', color = 'Red')
                    plt.title('MSE: %s' %MSE)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                
                return MSE
        
        else:
            # realizando a previsao com o sensor passado
            prediction = self.sensores[num_sensor].Predizer(Entradas)
            return prediction
        
    def Realizar_Previsao(self, Entradas):
        '''
        Metodo para realizar a previsao com a melhor particula (ELM) do enxame
        :param Entradas: padroes de entrada para realizar a previsao
        :return: Retorna a predicao para as entradas apresentadas
        '''
        
        return self.best_elm.Predizer(Entradas)
             
    def Treinar(self):
        '''
        Metodo para treinar a rede ELM com o IDPSO 
        '''
        
        self.Criar_Particula()       
        
        i = 0
        while(i < self.iteracoes):
            i = i + 1
            
            self.Fitness()
            self.Gbest()
            self.Pbest()
            self.Velocidade()
            #self.Atualizar_parametros(i)
            self.Atualizar_particulas()
            i = self.Criterio_parada(i)
            
            #print("[%d]" % (i) + " : ", self.gbest.fitness)
            #self.Grafico_Convergencia(self.gbest.fitness, i)
        
        self.Melhor_ELM()
            
    def Melhor_ELM(self):
        '''
        método para retornar o melhor ELM
        '''
        
        ELM = ELMRegressor(self.qtd_neuronios)
        posicao = self.gbest.posicao.reshape(self.linhas, self.qtd_neuronios)
        ELM.Treinar(self.dataset[0], self.dataset[1], posicao)
            
        self.best_elm = ELM
        
    def Espalhar_particulas(self):
        '''
        método para apagar as informações de parte do enxame, essa parte é definida por uma porcentagem: self.tx_espalhar
        '''
        
        self.best_elm = []
        
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
            p.velocidade = array([0.0 for i in range(self.numero_dimensoes)])
            p.best = p.posicao
            p.fit_best = p.fitness
            p.c1 = self.c1
            p.c2 = self.c2
            p.inercia = self.inercia
            p.phi = 0
            self.particulas.append(p)
            
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
            #self.Atualizar_parametros(i)
            self.Atualizar_particulas()
            
            i = self.Criterio_parada(i)
            #print("[%d]" % (i) + " : ", self.gbest.fitness)
            #self.Grafico_Convergencia(self.gbest.fitness, i)
            
        self.Melhor_ELM()
       
def main():
    print()
    
if __name__ == "__main__":
    main()
    


