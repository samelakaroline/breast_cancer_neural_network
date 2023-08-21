# -*-coding:utf8 -*-

import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torchvision
from scipy import optimize


#Base de dados

#definir seeds para a geração de números aleatórios 
np.random.seed(123)
torch.manual_seed(123)

#Variáveis para chamar os arquivos csv

previsores = pd.read_csv('C:\\Users\\Sâmela Rocha\\Documents\\git\\Projeto 1 - Classificação Binária bred cancer\\projeto1\\entradas-breast.csv')
classe = pd.read_csv('C:\\Users\\Sâmela Rocha\\Documents\\git\\Projeto 1 - Classificação Binária bred cancer\\projeto1\\saidas-breast.csv')

#divisão de teste e treinamento

previsores_teste, previsores_treinamento, classe_teste, classe_treinamento = train_test_split(previsores,
                                                                                              classe,
                                                                                              test_size=0.25) #test_size, indica que 25% dos dados serão usados para teste

#transformar dados para tensores

previsores_treinamento = torch.tensor(np.array(previsores_treinamento), dtype = torch.float)
classe_treinamento = torch.tensor(np.array(classe_treinamento), dtype = torch.float)

#criar dataset de treinamento

dataset = torch.utils.data.TensorDataset(previsores_treinamento, classe_treinamento)
train_loader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)

#Definindo Modelo

classificador = nn.Sequential(
    nn.Linear(in_features=30, out_features=16),
    nn.ReLU(),
    nn.Linear(16,16),
    nn.ReLU(),
    nn.Linear(16,1),
    nn.Sigmoid()
)

#função de erro

criterion = nn.BCELoss()

#função de otimização

optimizer = torch.optim.Adam(classificador.parameters(), lr= 0.001, weight_decay=0.0001)
#Treinamento

for epoch in range(100):
    running_loss = 0.
    for data in train_loader:
        inputs, labels = data
        #print(inputs)
        #print('---------')
        #print(labels)
        optimizer.zero_grad()

        outputs = classificador.forward(inputs)
        #print(outputs)
        loss = criterion(outputs,labels)
        #print(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Época %3d: perda %.5f' % (epoch+1, running_loss/len(train_loader)))
        
#colocar a rede em modo de avaliação para que não continue o treinamento 

previsores_teste = torch.tensor(np.array(previsores_teste), dtype=torch.float)

previsoes = classificador.forward(previsores_teste)

previsoes = np.array(previsoes > 0.5)

print (previsoes) #respostas da rede neural depois de treinada

print(classe_teste)

taxa_acerto = accuracy_score(classe_teste, previsoes)

matriz = confusion_matrix(classe_teste, previsoes)

print (matriz)

sns.heatmap(matriz, annot=True)

print(taxa_acerto)