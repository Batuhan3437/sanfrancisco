import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("reading_data/train.csv")
#print(df)

#enlem boylam dates verileri işime yarıyacak daha hızlı fit(eğitebilmek) edebilmek için drop ediyoruz
#axis=1 column drop yani sütun drop için kullanılıyor.axis=0 Row drop yani satır için kullanılıyor.
df=df.drop(['PdDistrict','Address','Resolution','Descript','DayOfWeek'],axis=1)

df.isnull().sum()
#hiç null değer yok

#bu lambda fonksiyon işlemi dates deki saat ve tarihi ayırıyor ve sadece tarih kısmını alıyor
f= lambda x: (x["Dates"].split())[0]
df["Dates"]=df.apply(f,axis=1)
#df.head()

#Burdada ay ve günden kurtuluyoruz tarih bilgisinde

f= lambda x: (x["Dates"].split('-'))[0]
df["Dates"]=df.apply(f,axis=1)
#df.head()

#bu seferde sadece 2014 olanları almak için bir lambda fonksiyonu yazıcaz
df_2014=df[df["Dates"]=='2014']
#df_2014.head()


#scale edicez değerleri küçültücez
scaler=MinMaxScaler()

#yeni sütun oluşturuyoruz X_scaled ve Y_scaled
scaler.fit(df_2014[['X']])
df_2014['X_scaled']=scaler.transform(df_2014[['X']])


scaler.fit(df_2014[['Y']])
df_2014['Y_scaled']=scaler.transform(df_2014[['Y']])


#kaç tane cluster küme kullanacağımızı belirlemek için elbow metodunu kullanıyrouz.
k_range=range(1,15)

list_dist=[]

for k in k_range:
    model=KMeans(n_clusters=k)
    model.fit(df_2014[['X_scaled','Y_scaled']])
    list_dist.append(model.inertia_)
    
#tablo oluşturma dirsek değerini bulmak için yani K yı 
from matplotlib import pyplot as plt

plt.xlabel('K')
plt.ylabel('Distortion value (intertia)')
plt.plot(k_range,list_dist)
plt.show()
#Dirsek noktası K=5 te

#suç kayıtlarına kümeye ayırıyoruz konuma göre
model=KMeans(n_clusters=5)
y_predicted=model.fit_predict(df_2014[['X_scaled','Y_scaled']])
#print y_predicted
df_2014['cluster']=y_predicted
#print df_2014

