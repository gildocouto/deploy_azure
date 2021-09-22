import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template, url_for

import re, pickle, nltk, unicodedata, glob

from datetime import date, datetime
from collections import Counter
import matplotlib.pyplot as plt

from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from stop_words import get_stop_words
import spacy
from spacy.lang.pt import Portuguese
from spacy.lang.pt.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Instanciar o Flask
app = Flask(__name__)

# Importar modelo do disco
modeloImportado = pickle.load(open('randomForestRegressor.sav','rb'))

#Importando vocabulário do modelo
vocabulario = pickle.load(open(vocabulario_modelo.txt,'rb'))

#Criando objeto vetorizador "vetorizador_tfidf"
vetorizador_tfidf = TfidfVectorizer(ngram_range=(1, 7),use_idf=True,vocabulary=vocabulario)
    
# end-point do flask
# Essa rota (route) vai redirecionar para a função abaixo. Toda que vez que acessar esse end-point essa função (predict) vai rodar
@app.route('/predicao',methods=['POST'])

def predicao():

    dic_classe_assunto = {
    0 : 'CONTRIBUIÇÕES DE TERCEIROS_Base de cálculo (salário de contribuição). Limite máximo de 20 (vinte) vezes o salário mínimo (art. 4º da Lei 6.950/81)._TRIBUTOS E CONTRIBUIÇÕES ADMINISTRADOS',
    1 : 'PIS/COFINS_Base de cálculo. Exclusão do ICMS._TRIBUTOS E CONTRIBUIÇÕES ADMINISTRADOS',
    2 : 'CONTRIBUIÇÃO PREVIDENCIÁRIA_Lei nº 8.212/91. Não exigência sobre salário-maternidade, auxílio-doença, auxílio-creche, vale-transporte, auxílio-acidente, abono de férias, férias, adicional de férias de um terço e/ou aviso prévio._TRIBUTOS E CONTRIBUIÇÕES ADMINISTRADOS',
    3 : 'CONTRIBUIÇÕES DE TERCEIROS_Salário-educação, Sistema "S" e Incra. Inconstitucionalidade do recolhimento sobre a folha de salários._TRIBUTOS E CONTRIBUIÇÕES ADMINISTRADOS',
    4 : 'PIS/COFINS_Base de cálculo. Exclusão do PIS/Cofins - Cálculo por dentro._TRIBUTOS E CONTRIBUIÇÕES ADMINISTRADOS',
    5 : 'PIS/COFINS_Base de cálculo. Exclusão do ICMS e/ou ISS e do próprio PIS/Cofins - Cálculo por dentro._TRIBUTOS E CONTRIBUIÇÕES ADMINISTRADOS',
    6 : 'CSLL/IRPJ_Não incidência sobre valores recebidos a título de Selic na repetição do indébito._TRIBUTOS E CONTRIBUIÇÕES ADMINISTRADOS'}

    data = request.get_json(force=True)

    #Coletor de dados
    if data: #Saber se retornou algum dados, se não é vazio.
        textos = data['texto']

        ### Preparação dos dados (acrescentar código de preparação de dados)
        
        #Retirando pontuações, caracteres especiais e transformando texto em minúsculo
        textos = ''.join([re.sub(r'[^\w\s]', ' ', palavra).lower() for palavra in textos])

        #Aplicando Steeming (redução palavras ao radical) e removendo palavras com dois ou menos caracteres
        stemmer = nltk.stem.RSLPStemmer()
        textos = ' '.join([stemmer.stem(palavra) for palavra in textos.split()
                          if len(palavra)>2])

        #Retornando texto após steeming sem acentos
        texto_processado = unicodedata.normalize(u'NFKD', textos).encode('ascii', 'ignore').decode('utf8')

        total_palavras = len(texto_processado)
        #contagem_dic = Counter(texto_processado.split())

        if total_palavras != 0:
            #Classe predição
            texto_processado_array_vetor = vetorizador_tfidf.fit_transform(np.array([texto_processado])).toarray()

            classe_predicao = modeloImportado.predict(texto_processado_array_vetor)[0]
            assunto_predicao = dic_classe_assunto[classe_predicao]
            probabilidade_classes = modeloImportado.predict_proba(texto_processado_array_vetor)
            documentos_vazios = None    
    
        else:
            classe_predicao = 'PDF não pesquisável'
            assunto_predicao = 'PDF não pesquisável'
            probabilidade_classes = 'PDF não pesquisável'
            documentos_vazios = 'PDF não pesquisável'
    
    
    else:
            classe_predicao = 'Documento txt vazio'
            assunto_predicao = 'Documento txt vazio'
            probabilidade_classes = 'Documento txt vazio'
            documentos_vazios = 'Documento txt vazio'
        
    return jsonify(classe_predicao)

if __name__ == '__main__':
    # Iniciar Flask
    app.run(debug=True)