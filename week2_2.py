import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

dpath = 'Titanic-Dataset-fixed.csv' #membuat objek yang berisi 'Titanic-Dataset.csv'
df = pd.read_csv(dpath) #mengambil data dengan menggunakan read_scv pada variabel data
df.head() #mengambil 5 data teratas
print(df.head()) #menampilkan data dari df.head

#langkah2
df = df[['Survived', 'Pclass', 'Age', 'Sex', 'Cabin']] #menampilkan data dengan kolom yang dipilih
df.head() #mengambil 5 data teratas
print(df.head()) #menampilkan data dari df.head

#langkah3
le = LabelEncoder() #membuat objek dari LabelEncoder
df['Sex'] = le.fit_transform(df['Sex']) #encoding kolom sex
df['Cabin'] = le.fit_transform(df['Cabin']) #encoding kolom cabin

#langkah4
df.head() #mengambil 5 data teratas
print(df.head()) #menampilkan data dari df.head

#langkah5
std = StandardScaler() #membuat objek dari StandardScaler
df['Age'] = std.fit_transform(df[['Age']]) #proses standarisasi

#langkah6
df.head() #mengambil 5 data teratas
print(df.head()) #menampilkan data dari df.head