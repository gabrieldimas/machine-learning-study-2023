#langkah1
import pandas as pd

data = 'Titanic-Dataset.csv' #membuat objek yang berisi 'Titanic-Dataset.csv'
df = pd.read_csv(data) #mengambil data dengan menggunakan read_scv pada variabel data
df.head() #menampilkan 5 data teratas
print(df.head())

#langkah2
df.info() #melihat struktur data

null_data = df.isnull().sum() #mengambil jumlah data yang hilang pada tiap kolom
print(null_data) #menampilkan total data yang hilang

#langkah3
df['Age'].fillna(value=df['Age'].mean(), inplace=True) #mengganti data null pada kolom Age menjadi nilai yang diambil dari mean kolom
df['Cabin'].fillna(value="DECK", inplace=True) #mengganti data null pada kolom cabin menjadi data "DECK"
df['Embarked'].fillna(value=df['Embarked'].mode, inplace=True) #mengganti data null pada kolom embarked menjadi nilai yang diambil dari modus kolom

df.info() #menampilkan struktur data, disini bisa kita lihat bahwa data non-null sama menyimbolkan tidak ada data null