#Importando os pacotes necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
import numpy as np

#Lendo o arquivo txt
fruits = pd.read_table('Dataset/fruit_data_with_colors.txt')

#Mostrando o head do arquivo carregado
fruits.head()

#Sumário estatístico
fruits.describe()

#Quantidade de linhas frutas e quantas colunas existem no arquivo 
print(fruits.shape)

#Tipo dos nomes de frutas únicos
print(fruits['fruit_name'].unique())

#Quantidade dos nomes de frutas
print(fruits.groupby('fruit_name').size())

#Plotando em um gráfico Nome por quantidade
sns.countplot(fruits['fruit_name'],label="Count")
plt.show()

#Box plot
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 
title='Box plot para cada váriavel')
plt.savefig('fruits_box')
plt.show()

# Grafico historgrama
fruits.drop('fruit_label' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histograma para cada váriavel númerica")
plt.savefig('fruits_hist')
plt.show()
#Podemos visualizar que color_score é muito parecido com uma distribuião normal ou Gaussiana


from pandas.tools.plotting import scatter_matrix
from matplotlib import cm
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']
cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fruits_scatter_matrix')

#Podemos analisar que algumas variáveis são bastante correlacionadas e tem uma relação estatística de predição, verificar (mass x width)


#Treino e Teste aplicando scalling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#SVM
#Support vector machine

from sklearn.svm import SVC
svm = SVC(gamma='scale')
svm.fit(X_train, y_train)
print('Acurácia de SVM na base de treino {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Acurácia de SVM na base de teste {:.2f}'
     .format(svm.score(X_test, y_test)))

#Report de classificação
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

#Matriz de confunsão
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))


#Visualizar
from matplotlib.colors import ListedColormap
plt.figure("Training Set")
x_set ,y_set = x_train,y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01 ),
                    np.arange(start = x_set[:,0].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))
plt.contourf(x1,x2,nb_classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.8, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow'))(i), label = j)
plt.title(' Naive Baise Classifier (Training set)')
plt.xlabel('red (not bought)')
plt.ylabel('green (bought)')
plt.legend()
plt.show()

plot_fruit_knn(X_train, y_train, 5, 'uniform')

plot_fruit_knn(X_train, y_train, 1, 'uniform')

plot_fruit_knn(X_train, y_train, 10, 'uniform')

plot_fruit_knn(X_test, y_test, 5, 'uniform')
k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])


#parameters of following function are mass,width and height
#example1
prediction1=knn.predict([['10','6.3','8']])
predct[prediction1[0]]

#example2
prediction2=knn.predict([['300','7','10']])
predct[prediction2[0]]