#-*- coding: utf-8 -*-
'''
Created on 31 de mar de 2017

@author: gusta
'''
from numpy import random as rd
import numpy as np 
import matplotlib.pyplot as plt
from geradores_tabela.Tabela_excel import Tabela_excel
from numpy import array
from numpy.random.mtrand import uniform
import pandas as pd

caminho_salvar = 'series/Series Geradas/'

# valor para nao transbordar a geracao de conceitos
valor = 0.5

class Gerador_conceitos():
    def __init__(self):
        '''
        Classe para instanciar o gerador de conceitos 
        '''
        
        self.inicio = False
        self.serie_dividida = []
        self.serie_final = []
        self.t = 1
        self.conceito_atual = 0
        self.contador_continuo = 0
        self.t_continuo = 1
        self.serie_continua = []
        self.ativo = True
         
    def Increment_Add(self, vetor, valor):
        '''
        Metodo para inserir mais dados na janela deslizante 
        :param vetor: vetor que tera o valor incrementado
        :param valor: valor de entrada
        '''

        aux = [0.0] * (len(vetor)+1)
        aux = np.asarray(aux)
        aux[:len(vetor)] = vetor
        aux[len(vetor)] = valor
        vetor = aux
        
        #sabendo em que tempo a serie está
        self.t += 1
        
        return vetor
        
    def Fila_Add(self, valor):
        '''
        Metodo para inserir um valor em uma janela deslizante, o valor mais antigo sera excluido 
        :param valor: valor de entrada
        '''
        aux = [0.0] * len(self.observacoes)
        aux = np.asarray(aux)
        aux[len(self.observacoes)-1] = valor
        aux[:len(aux)-1] = self.observacoes[1:]
        self.observacoes = aux
            
    def Conferir_dados(self, parametros, observacoes_iniciais = None, equacao4 = None):
        '''
        Metodo para conferir se os parametros passados nos metodos estão corretos 
        :param parametros: vetor com o conjunto de parametros para os modelos
        :param observacoes_iniciais: vetor com o conjunto de valores iniciais para serie
        '''
        
        if(self.inicio == False):
            if(equacao4 == True):
                if(observacoes_iniciais.any()):
                    self.inicio = True
                    p = 2
                    if(len(observacoes_iniciais) != p):
                        print("Conjunto de valores iniciais nao condiz com o tamanho da ordem p da equação 4")
                    else:
                        self.observacoes = observacoes_iniciais
                else:
                    print("Insira os valores das observações iniciais!")
                
            else:
                if(observacoes_iniciais.any()):
                    self.inicio = True
                    p = len(parametros)
                    if(len(observacoes_iniciais) != p):
                        print("Conjunto de valores iniciais nao condiz com o tamanho da ordem: p.")
                    else:
                        self.observacoes = observacoes_iniciais
                else:
                    print("Insira os valores das observações iniciais!")
                
        else:
            if(equacao4 == True):
                p = 2
                aux = len(self.serie_dividida)-1
                lista = self.serie_dividida[aux]
                self.observacoes = lista[len(lista)-p:]
            else:
                p = len(parametros)
                aux = len(self.serie_dividida)-1
                lista = self.serie_dividida[aux]
                self.observacoes = lista[len(lista)-p:] 

    def Equacao_1(self, parametros, variancia):
        '''
        essa equacao gera valores para um modelo AR(p)
        :param parametros: parametros para o modelo ar
        :param variancia: variancia para o ruido branco
        :return um dado para uma sequencia no instante t:
        '''
        x = 0
        for i in range(len(parametros)):
            x += parametros[i] * self.observacoes[i]
            #print("x: " +str(x)+ " = " +str(parametros[i])+ " * " +str(self.observacoes[i]))
        w = rd.normal(0, variancia, 1)
        #print("w: " + str(w))
        x = x+w
        #print("x: " +str(x))
        return x
    
    def Equacao_2(self, tendencia, beta_vetor, variancia_ruido):
        '''
        essa equacao gera dados para um modelo linear sazonal
        :param tendencia: componente de tendencia para os valores
        :param beta_vetor: vetor com os dados da sazonalidade
        :param variancia_ruido: variancia para o ruido branco
        :return um dado para uma sequencia no instante t:
        '''
       
        m = tendencia
        s = len(beta_vetor)
        
        resto = ((self.t-1) % s)
        if(resto >= s-1):
            resto = s-2

        indice_beta = 1 + resto
        beta = beta_vetor[indice_beta]
        
        w = rd.normal(0, variancia_ruido, 1)
       
        x = m + beta + w
        
        return x
    
    def Equacao_3(self, parametros, variancia):
        '''
        essa equacao gera valores para um modelo não linear de ordem p
        :param parametros: parametros para o modelo não linear
        :param variancia: variancia para o ruido branco
        :return um dado para uma sequencia no instante t:
        '''
        x = 0
        for i in range(len(parametros)):
            x += parametros[i] * self.observacoes[i]
        
        parte1 = -10*self.observacoes[0]
        parte2 = np.exp(parte1)
        formula = (1 - parte2)
        formula = formula**-1
        
        w = rd.normal(0, variancia, 1)
        
        x = (x * formula) + w
        
        if(x-self.observacoes[-1] >= valor):
            x = self.observacoes[-1] - w
        
        if(x-self.observacoes[-1] <= -valor):
            x = -self.observacoes[-1] + w
            
        return x
    
    def Equacao_4(self, parametros, variancia):
        '''
        essa equacao gera valores para um modelo não linear com duas observacoes
        :param parametros: vetor com 4 posicoes com os parametros para o modelo não linear
        :param variancia: variancia para o ruido branco
        :return um dado para uma sequencia no instante t:
        '''
        
        if(self.observacoes[0] == 0):
            self.observacoes[0] = 0
            
        if(self.observacoes[1] == 0):
            self.observacoes[1] = 0
            
            
        x = parametros[0] * self.observacoes[1] + parametros[1] * self.observacoes[0] + parametros[2] * self.observacoes[1] + parametros[3] * self.observacoes[0]
        
        formula = (1 - np.exp(-10*self.observacoes[0]))
        formula = formula**-1
        
        w = rd.normal(0, variancia, 1)
        
        x = (x * formula) + w
        
        if(x >= valor):
            x = valor + w
        
        if(x <= -valor):
            x = -valor - w
              
        return x
    
    def modelo_AR(self, parametros, variancia, tamanho_do_conceito, observacoes_iniciais = None):
        '''
        esse metodo gera um conceito com caracteristicas de uma serie linear feita por um modelo AR(p) 
        :param parametros: vetor com os parametros para o modelo ar
        :param variancia: variancia para o ruido branco
        :param tamanho_do_conceito: tamanho do conceito a ser gerado
        :param observacoes_iniciais: se for o primeiro conceito a ser gerado, deve se especificar as observacoes iniciais
        '''
        
        self.Conferir_dados(parametros, observacoes_iniciais)
            
        serie = []
        
        for i in range(tamanho_do_conceito):
            x = self.Equacao_1(parametros, variancia)
            serie = self.Increment_Add(serie, x)
            self.Fila_Add(x)
        self.serie_dividida.append(serie)
        
    def modelo_sazonal(self, tendencia, beta_vetor, variancia, tamanho_do_conceito, observacoes_iniciais = None):
        '''
        esse metodo gera um conceito com caracteristicas de um modelo linear sazonal
        :param tendencia: componente de tendencia para os valores
        :param beta_vetor: vetor com os dados da sazonalidade
        :param variancia: variancia para o ruido branco
        :param tamanho_do_conceito: tamanho do conceito a ser gerado
        :param observacoes_iniciais: se for o primeiro conceito a ser gerado, deve se especificar as observacoes iniciais
        '''
        
        self.Conferir_dados(beta_vetor, observacoes_iniciais)
            
        serie = []
        
        for i in range(tamanho_do_conceito):
            x = self.Equacao_2(tendencia, beta_vetor, variancia)
            serie = self.Increment_Add(serie, x)
            self.Fila_Add(x)
        self.serie_dividida.append(serie)
        
    def modelo_nlinear1(self, parametros, variancia, tamanho_do_conceito, observacoes_iniciais = None):
        '''
        esse metodo gera um conceito com caracteristicas de uma serie não linear feita pela equacao 3 
        :param parametros: vetor com os parametros para o modelo 
        :param variancia: variancia para o ruido branco
        :param tamanho_do_conceito: tamanho do conceito a ser gerado
        :param observacoes_iniciais: se for o primeiro conceito a ser gerado, deve se especificar as observacoes iniciais
        '''
        
        self.Conferir_dados(parametros, observacoes_iniciais)
            
        serie = []
        
        for i in range(tamanho_do_conceito):
            x = self.Equacao_3(parametros, variancia)
            serie = self.Increment_Add(serie, x)
            self.Fila_Add(x)
            #print(self.observacoes)
        self.serie_dividida.append(serie)
        
    def modelo_nlinear2(self, parametros, variancia, tamanho_do_conceito, observacoes_iniciais = None):
        '''
        esse metodo gera um conceito com caracteristicas de uma serie não linear feita pela equacao 4 
        :param parametros: vetor com os parametros para o modelo 
        :param variancia: variancia para o ruido branco
        :param tamanho_do_conceito: tamanho do conceito a ser gerado
        :param observacoes_iniciais: se for o primeiro conceito a ser gerado, deve se especificar as observacoes iniciais
        '''
        
        self.Conferir_dados(parametros, observacoes_iniciais, equacao4=True)
            
        serie = []
        
        for i in range(tamanho_do_conceito):
            x = self.Equacao_4(parametros, variancia)
            serie = self.Increment_Add(serie, x)
            self.Fila_Add(x)
        self.serie_dividida.append(serie)
        
    def Obter_Serie(self):
        '''
        esse metodo empilha todos os conceitos gerados
        :return: retorna a serie completa com todos os conceitos
        '''
        
        tamanho_serie = 0
        self.qtd_conceitos = 0
        
        for i in self.serie_dividida:
            tamanho_serie += len(i)
            self.qtd_conceitos += 1
            
        serie = [0] * tamanho_serie
        
        for x, i in enumerate(self.serie_dividida):
            if(x == 0):
                inicio = 0
                serie[inicio:len(i)] = i
                inicio += len(i)
            else:
                serie[inicio:inicio+len(i)] = i
                inicio += len(i)
        
        self.serie_final = serie
        
        return self.serie_final
    
    def Gerador_continuo(self, proximo_conceito):
        '''
        esse gera dados continuamente com os conceitos definidos anteriomente
        :param proximo_conceito: variavel booleana para informar que deve-se mudar de conceito
        :return: retorna o valor para o instante de tempo t
        '''

        if(proximo_conceito == True):
            
            self.contador_continuo = 0
            self.conceito_atual += 1
            self.qtd_conceitos = len(self.serie_dividida)
            
            if(self.conceito_atual == self.qtd_conceitos):
                self.ativo = False
                return
            else:    
                valor = self.serie_dividida[self.conceito_atual][self.contador_continuo-1]
        else:
            self.contador_continuo += 1
            valor = self.serie_dividida[self.conceito_atual][self.contador_continuo-1]
        
        
        print("Conceito atual: [", self.conceito_atual, "][", self.t_continuo, "]: ", valor)

        self.t_continuo += 1
        self.serie_continua = self.Increment_Add(self.serie_continua, valor) 
   
        return valor
                
    def Plotar(self, serie = None, deteccoes = None):
        '''
        esse metodo plota a serie final com todos os conceitos divididos por uma reta vermelha
        '''
        
        if(serie == None):
            plt.plot(self.serie_final, label = 'Série', color = 'Blue')
            
            contador = len(self.serie_dividida[0])
            for i in range(len(self.serie_dividida)):
                plt.axvline(contador, linewidth=1, color='r', zorder=-1)
                
                if(i > 0):
                    contador += len(self.serie_dividida[i])
                
            plt.title('Série com %s conceitos' %(self.qtd_conceitos))
            plt.legend()
            plt.tight_layout()
            #plt.axis([0, 10000, -50, 50])
            plt.show()  
        else:  
            plt.plot(serie, label = 'Série', color = 'Blue')
            
            contador = deteccoes[0]
            for i in range(len(deteccoes)):
                plt.axvline(contador, linewidth=1, color='r', zorder=-1)
                
                if(i > 0):
                    contador = deteccoes[i]
                
            plt.tight_layout()
            plt.show()
    
    def Escrever_serie_xls(self, pasta, nome, serie):
        '''
        método para escrever a serie gerada em formato xls
        :param: pasta: pasta de destino
        :param: nome: nome do arquivo
        :param: serie: serie temporal que sera escrita 
        '''
        
        tabela = Tabela_excel()
        caminho = caminho_salvar+pasta+nome
        folhas = [nome]
        largura_col = 5000
        tabela.Criar_tabela(caminho, folhas, largura_col=largura_col)
        
        for x, i in enumerate(serie):
            tabela.Adicionar_dado(0, x, 0, i)
    
    def Escrever_serie_csv(self, pasta, nome, serie):
        '''
        método para escrever a serie gerada em formato csv
        :param: pasta: pasta de destino
        :param: nome: nome do arquivo
        :param: serie: serie temporal que sera escrita 
        '''

        caminho = caminho_salvar+pasta+nome

        df = pd.DataFrame({'Serie' : serie})
    
        df.to_csv(caminho+".csv", header=False, index=False, float_format='%.3f')
        
        del df

    def series_lineares_ictai(self, tamanho_conceitos, qtd_series, grafico):
        '''
        método para criar as series lineares feitas no artigo ICTAI
        :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
        :param: qtd_series: quantidade de series que serão criadas
        :param: grafico: variavel booleana para plotar apos a criacao da serie 
        '''

        observacoes_iniciais = array([uniform(-2, 2) for i in range(3)])
        variancia = 0.1
        
        self.modelo_AR(parametros=[0.42, 0.28, 0.005], variancia=variancia, tamanho_do_conceito=tamanho_conceitos, observacoes_iniciais=observacoes_iniciais)
        #self.modelo_AR(parametros=[0.42, 0.28, 0.005], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.003, -0.005, 1.0], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.018, 0.95, 0.032], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.11, 0.32, -0.028, 0.038, 0.48], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.032, 0.41, -0.24, -0.22, 1.0], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.14, -0.29, 0.0025, 1.0], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.032, 0.41, -0.24, -0.22, 1.0], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.018, 0.95, 0.032], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.003, -0.005, 1.0], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.42, 0.28, 0.005], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        
        serie = self.Obter_Serie()
        nome = "lin-" 
        pasta = "lineares/"
        
        if(grafico == True):
            self.Plotar()
            
        return [pasta, nome, serie]

    def series_nlineares_ictai(self, tamanho_conceitos, qtd_series, grafico):
        '''
        método para criar as series não lineares feitas no artigo ICTAI
        :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
        :param: qtd_series: quantidade de series que serão criadas
        :param: grafico: variavel booleana para plotar apos a criacao da serie 
        '''
        
        observacoes_iniciais = array([uniform(-2, 2) for i in range(4)])
        
        variancia = 0.1
        
        self.modelo_nlinear1(parametros=[0.55, 0.024, 0.41, 0.009], variancia=variancia, tamanho_do_conceito=tamanho_conceitos, observacoes_iniciais=observacoes_iniciais)
        #self.modelo_nlinear1(parametros=[0.55, 0.024, 0.41, 0.009], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear2(parametros=[0.059, 0.086, 0.62, 0.21], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.47, 0.57, 0.14, -0.19], variancia=0.3, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.55, 0.024, 0.41, 0.009], variancia=0.4, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear2(parametros=[0.059, 0.086, 0.62, 0.21], variancia=0.4, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear2(parametros=[0.55, 1.0, 0.0028, -0.58], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear2(parametros=[0.059, 0.086, 0.62, 0.21], variancia=0.4, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.55, 0.024, 0.41, 0.009], variancia=0.4, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.47, 0.57, 0.14, -0.19], variancia=0.3, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear2(parametros=[0.059, 0.086, 0.62, 0.21], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        
        serie = self.Obter_Serie()
        pasta = "nlineares/"
        nome = "nlin-"
        
        if(grafico == True):
            self.Plotar()
            
        return [pasta, nome, serie]
    
    def series_sazonais_ictai(self, tamanho_conceitos, qtd_series, grafico):
        '''
        método para criar as series sazonais feitas no artigo ICTAI
        :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
        :param: qtd_series: quantidade de series que serão criadas
        :param: grafico: variavel booleana para plotar apos a criacao da serie 
        '''
        
        variancia = 3.5
        
        self.modelo_sazonal(tendencia=0, beta_vetor=[34, 32, 30, 28, 26, 24, 22, 24, 26, 28, 30, 32], variancia=variancia, tamanho_do_conceito=tamanho_conceitos, observacoes_iniciais=[34, 32, 30, 28, 26, 24, 22, 24, 26, 28, 30, 32])
        #self.modelo_sazonal(tendencia=0, beta_vetor=[34, 32, 30, 28, 26, 24, 22, 24, 26, 28, 30, 32], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[34, 26, 18, 10, 18, 26], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[34,26,18,10,18,26], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[34, 26, 18, 10, 2, -6, -14, -6, 2, 10, 18, 26], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[34, 10, -14, 10], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[38, 28, 18, 8, 0, -8,-18,-8, 0, 8, 18, 28], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[34, 10,-14, 10], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[34, 26, 18, 10, 2, -6, -14, -6, 2, 10, 18, 26], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[34,26,18,10,18,26], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=0, beta_vetor=[34, 26, 18, 10, 18, 26], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        
        serie = self.Obter_Serie()
        pasta = "sazonais/"
        nome = "saz-"
        
        if(grafico == True):
            self.Plotar()
        
        return [pasta, nome, serie]    
            
    def series_hibridas_ictai(self, tamanho_conceitos, qtd_series, grafico): 
        '''
        método para criar as series hibridas feitas no artigo ICTAI
        :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
        :param: qtd_series: quantidade de series que serão criadas
        :param: grafico: variavel booleana para plotar apos a criacao da serie 
        '''
        
        variancia = 0.1
        
        observacoes_iniciais = array([uniform(-2, 2) for i in range(3)])             
        self.modelo_AR(parametros=[0.003, -0.005, 1.0], variancia=variancia, tamanho_do_conceito=tamanho_conceitos, observacoes_iniciais= observacoes_iniciais)
        #self.modelo_AR(parametros=[0.003, -0.005, 1.0], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=1, beta_vetor=self.observacoes, variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear2(parametros=[0.059, 0.086, 0.62, 0.21], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.018, 0.95, 0.032], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=2, beta_vetor=self.observacoes, variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.55, 0.024, 0.41, 0.009], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.018, 0.95, 0.032], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_sazonal(tendencia=2, beta_vetor=self.observacoes, variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear2(parametros=[0.059, 0.086, 0.62, 0.21], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.55, 0.024, 0.41, 0.009], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
            
        serie = self.Obter_Serie()
        pasta = "hibridas/"
        nome = "hib-"
        
        if(grafico == True):
            self.Plotar()
        
        return [pasta, nome, serie]
    
    def series_lineares_graduais_revista(self, tamanho_conceitos, qtd_series, grafico):
        '''
        método para criar as series lineares feitas no artigo ICTAI
        :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
        :param: qtd_series: quantidade de series que serão criadas
        :param: grafico: variavel booleana para plotar apos a criacao da serie 
        '''

        observacoes_iniciais = array([uniform(0, 0.5) for i in range(4)])
        variancia = 0.02
        
        self.modelo_AR(parametros=[0.006607488803146307, -0.2529881594354167, 0.8552562304577513, 0.3905674250981309], variancia=variancia, tamanho_do_conceito=tamanho_conceitos, observacoes_iniciais=observacoes_iniciais)
        self.modelo_AR(parametros=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.00301790735789223, -0.3277435418893056, 0.14635639512590287, 1.171740105825536], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.33266433992324546, -0.11265182345778371, 0.05425610414373307, 0.7151247572018152], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.6335599337412476, 0.334852076313965, 1.3595185287185048, -0.07363691675509275], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.44071503228306824, 0.07407529241129636, 1.2573688275751191, 0.1076310711909298], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.6335599337412476, 0.334852076313965, 1.3595185287185048, -0.07363691675509275], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.33266433992324546, -0.11265182345778371, 0.05425610414373307, 0.7151247572018152], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.00301790735789223, -0.3277435418893056, 0.14635639512590287, 1.171740105825536], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        
        serie = self.Obter_Serie()
        nome = "lin-grad" 
        pasta = "lineares_graduais/"
        
        if(grafico == True):
            self.Plotar()
            
        return [pasta, nome, serie]
    
    def series_lineares_abruptas_revista(self, tamanho_conceitos, qtd_series, grafico):
        '''
        método para criar as series lineares feitas no artigo ICTAI
        :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
        :param: qtd_series: quantidade de series que serão criadas
        :param: grafico: variavel booleana para plotar apos a criacao da serie 
        '''

        observacoes_iniciais = array([uniform(0, 0.5) for i in range(4)])
        variancia = 0.02
        
        self.modelo_AR(parametros=[0.14876092573738822, 0.05087244788237593, 0.4330193805067835, 0.3667339588762431], variancia=variancia, tamanho_do_conceito=tamanho_conceitos, observacoes_iniciais=observacoes_iniciais)
        self.modelo_AR(parametros=[-0.318229212036593, 0.4133521130815502, 1.14841972367221, -0.24486472090297637], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.00301790735789223, -0.3277435418893056, 0.14635639512590287, 1.171740105825536], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.026851477518947557, 0.22016898814054223, -0.03814933593273199, 0.8447046999175475], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.4785914515620302, 0.8558602481837317, 0.024539136949191378, 0.5980008075169353], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.026851477518947557, 0.22016898814054223, -0.03814933593273199, 0.8447046999175475], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[0.00301790735789223, -0.3277435418893056, 0.14635639512590287, 1.171740105825536], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_AR(parametros=[-0.318229212036593, 0.4133521130815502, 1.14841972367221, -0.24486472090297637], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        
        serie = self.Obter_Serie()
        nome = "lin-abt" 
        pasta = "lineares_abruptas/"
        
        if(grafico == True):
            self.Plotar()
            
        return [pasta, nome, serie]
    
    def series_nlineares_graduais_revista(self, tamanho_conceitos, qtd_series, grafico):
        '''
        método para criar as series lineares feitas no artigo ICTAI
        :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
        :param: qtd_series: quantidade de series que serão criadas
        :param: grafico: variavel booleana para plotar apos a criacao da serie 
        '''

        observacoes_iniciais = array([uniform(0, 0.5) for i in range(4)])
        variancia = 0.02
        
        self.modelo_nlinear1(parametros=[0.0203825939140348, 0.14856960377126693, 0.12154840218302701, 0.6913077037309644], variancia=variancia, tamanho_do_conceito=tamanho_conceitos, observacoes_iniciais=observacoes_iniciais)
        self.modelo_nlinear1(parametros=[0.21432208811179806, 0.1747177586312132, 0.25627781880181116, 0.34924372007037097], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.6747722679575656, 0.0400499490190765, 0.12859434021708172, 0.1411115580708043], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.2592127990366997, 0.18679044833178132, 0.2510160243812225, 0.29144511960870556], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.33266433992324546, -0.11265182345778371, 0.05425610414373307, 0.715124757201], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.1779748207049134, -0.09139762327444532, 0.3628849251594744, 0.5451838112044337], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.33266433992324546, -0.11265182345778371, 0.05425610414373307, 0.715124757201], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.2592127990366997, 0.18679044833178132, 0.2510160243812225, 0.29144511960870556], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.6747722679575656, 0.0400499490190765, 0.12859434021708172, 0.1411115580708043], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.21432208811179806, 0.1747177586312132, 0.25627781880181116, 0.34924372007037097], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        
        
        serie = self.Obter_Serie()
        nome = "nlin-grad" 
        pasta = "nlineares_graduais/"
        
        if(grafico == True):
            self.Plotar()
            
        return [pasta, nome, serie]
    
    def series_nlineares_abruptas_revista(self, tamanho_conceitos, qtd_series, grafico):
        '''
        método para criar as series lineares feitas no artigo ICTAI
        :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
        :param: qtd_series: quantidade de series que serão criadas
        :param: grafico: variavel booleana para plotar apos a criacao da serie 
        '''

        observacoes_iniciais = array([uniform(0, 0.5) for i in range(4)])
        variancia = 0.02
        
        self.modelo_nlinear1(parametros=[-0.06658679980732536, 0.23421353635081468, 0.15495114023325046, 0.6768101219541569], variancia=variancia, tamanho_do_conceito=tamanho_conceitos, observacoes_iniciais=observacoes_iniciais)
        self.modelo_nlinear1(parametros=[-0.506870130353138, 0.2589111765633722, 1.3970013340136547, -0.14964763809967313], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[-0.4387715888915295, 0.3747437070432394, 1.3330941335780706, -0.26908562619916504], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.06975366774909564, -0.05196107339800573, 0.6352865482608727, 0.3344985733604905], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[-0.2763541765783329, 0.3343598857377247, 0.4102952504128611, 0.5315753100371876], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[-0.4429405258368569, 0.4466229373805038, 1.351792157828681, -0.3561327432116702], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[0.06975366774909564, -0.05196107339800573, 0.6352865482608727, 0.3344985733604905], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[-0.506870130353138, 0.2589111765633722, 1.3970013340136547, -0.14964763809967313], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        self.modelo_nlinear1(parametros=[-0.06658679980732536, 0.23421353635081468, 0.15495114023325046, 0.6768101219541569], variancia=variancia, tamanho_do_conceito=tamanho_conceitos)
        
        serie = self.Obter_Serie()
        nome = "nlin-abt" 
        pasta = "nlineares_abruptas/"
        
        if(grafico == True):
            self.Plotar()
            
        return [pasta, nome, serie]
    
def Gerar_series(tipo, tamanho, qtd_series, grafico):
    '''
    método para gerar um conjunto dde series
    :param: tipo: tipo da serie a ser criada: 1- linear; 2- não-linear; 3-sazonal; 4- hibrida
    :param: tamanho_conceitos: tamanho das observacoes que cada conceito vai ter
    :param: qtd_series: quantidade de series que serão criadas
    :param: grafico: variavel booleana para plotar apos a criacao da serie 
    '''
    
    for i in range(qtd_series):
        
        gerador = Gerador_conceitos()
        
        print(str(i+1))
        
        if(tipo == 0):
            [pasta, nome, serie] = gerador.series_lineares_ictai(tamanho, qtd_series, grafico)
        elif(tipo == 1):
            [pasta, nome, serie] = gerador.series_nlineares_ictai(tamanho, qtd_series, grafico)
        elif(tipo == 2):
            [pasta, nome, serie] = gerador.series_sazonais_ictai(tamanho, qtd_series, grafico)
        elif(tipo == 3):
            [pasta, nome, serie] = gerador.series_hibridas_ictai(tamanho, qtd_series, grafico)
        elif(tipo == 4):
            print("Linear gradual")
            [pasta, nome, serie] = gerador.series_lineares_graduais_revista(tamanho, qtd_series, grafico)
        elif(tipo == 5):
            print("Linear abrupta")
            [pasta, nome, serie] = gerador.series_lineares_abruptas_revista(tamanho, qtd_series, grafico)
        elif(tipo == 6):
            print("Não linear gradual")
            [pasta, nome, serie] = gerador.series_nlineares_graduais_revista(tamanho, qtd_series, grafico)
        elif(tipo == 7):
            print("Não linear abrupta")
            [pasta, nome, serie] = gerador.series_nlineares_abruptas_revista(tamanho, qtd_series, grafico)
        #print(nome+str(i+1))
        
        gerador.Escrever_serie_csv(pasta, nome+str(i+1), serie)  
              
def main():
    ######################## Exemplo de como gerar e plotar o grafico #########################################################################################
    
    tamanho_conceitos = 2000
    qtd_series = 30
    grafico = False
    
    #Gerar_series(6, tamanho_conceitos, qtd_series, grafico)
     
    '''
    for i in range(4, 8):
        Gerar_series(i, tamanho_conceitos, qtd_series, grafico)
    '''
    
    '''
    dtst = Datasets()
    caminho = dtst.bases_reais(3)
    stream = pd.read_csv(caminho, header=None)
    
    nova = []
    for i in range(1, len(stream)-1):
        print(i)
        if(stream[2][i] != 'null'):
            nova.append(float(stream[2][i]))
    
    gerador = Gerador_conceitos()
    gerador.Escrever_serie_csv('../Series XLSX/Series Reais/', 'Dow', nova)
    
    plt.plot(nova)
    plt.show()
    '''
    
    
if __name__ == '__main__':
    main()
    

