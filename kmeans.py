# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 00:16:28 2020

@author: Mete
"""

#%%library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%create dataset
#gaussian veriler oluştur 1000 tane(25-5=20 25+5=30 20,30 arası değerler)
#class1
x1=np.random.normal(25,5,1000) 
y1=np.random.normal(25,5,1000)

#class 2
x2=np.random.normal(55,5,1000) 
y2=np.random.normal(60,5,1000)

#class3
x3=np.random.normal(55,5,1000) 
y3=np.random.normal(15,5,1000)

#x ve y leri yukarıdan aşağıya doğru birleştiriyoruz
x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)


dictionary={"x":x,"y":y}

data=pd.DataFrame(dictionary)


#data bilgileri için;
#data.info()
#data nın ortalaması standart sapması vb için;
#data.describe()

#%%Data görselleştirme
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()



#%% k_means algoritması bunu görecek

#plt.scatter(x1,y1,color="black")
#plt.scatter(x2,y2,color="black")
#plt.scatter(x3,y3,color="black")
#plt.show()

#%%KMeans
"""
Oluşan grafiğe göre en uygun k değerini bulup algoritma
adımlarına devam ediyoruz
"""
from sklearn.cluster import KMeans
wcss=[]

for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    #burda hangi değer maximum verimli onu ölçüyoruz
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.xlabel("number of k(cluster) value")
plt.ylabel("wcss")
plt.show()

#%%k=3 için modelleyelim
kmeans2=KMeans(n_clusters=3)
#datamızı sınıflara ayırıp predict ediyoruz ve nasıl bir
#görünüme kavuşağı hakkında güzel bir aşama kaydediyoruz
clusters=kmeans2.fit_predict(data)
#datamızla oluşan labeller kaç tane ise csv dosyamıza  ekliyoruz
data["label"]=clusters #data kontrol edebilirsiniz :)

#%%data görselleştirme
plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="green")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="blue")
#centroitler için ;)
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")
plt.show()



