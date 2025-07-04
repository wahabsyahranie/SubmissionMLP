"""IMPORT LIBRARY"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

"""LOAD DATASET"""
df = pd.read_csv('dataset/bank_transactions_data_edited.csv')

# Tampilkan 5 baris pertama dengan function head.
# print(df.head())

# Tinjau jumlah baris kolom dan jenis data dalam dataset dengan info.
# print(df.info())

# Menampilkan statistik deskriptif dataset dengan menjalankan describe
# print(df.describe())

# Menampilkan korelasi antar fitur (Opsional Skilled 1)
num_features = df.select_dtypes(include=[np.number])
# plt.figure(figsize=(9, 7))
# correlation_matriks = num_features.corr()
# sns.heatmap(correlation_matriks, annot=True,
#             cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Heatmap Korelasi')
# plt.show()
# dari hasil ini saya dapat melihat kolom yang cendrung berelasi adalah custmer age dan account balance, dimana semakin tua pelanggan kemungkinan semakin besar saldonya.

# # Menampilkan histogram untuk semua kolom numerik (Opsional Skilled 1)
# # 1. histogram LoginAttempts
# login1 = df.LoginAttempts[(df.LoginAttempts == 1)]
# login2_3 = df.LoginAttempts[(df.LoginAttempts > 1) & (df.LoginAttempts <= 3)]
# login3above = df.LoginAttempts[(df.LoginAttempts > 3)]

# x = ["1 (sukses)", "2-3 (mungkin lupa akun)", "3+ (indikasi bruteforce)",]
# y = [len(login1.values), len(login2_3.values), len(
#     login3above.values)]

# plt.figure(figsize=(15, 6))
# plt.bar(x, y, color=['blue', 'cyan', 'yellow'])
# plt.title("Percobaan login")
# plt.xlabel("Login")
# plt.ylabel("Number of Try")

# for i in range(len(x)):
#     plt.text(i, y[i], y[i], ha='center', va='bottom')
# plt.show()

# # 2. histogram CustomerAge
# age18_26 = df.CustomerAge[(df.CustomerAge >= 18) & (df.CustomerAge <= 26)]
# age27_44 = df.CustomerAge[(df.CustomerAge >= 27) & (df.CustomerAge <= 44)]
# age45_58 = df.CustomerAge[(df.CustomerAge >= 45) & (df.CustomerAge <= 58)]
# age58above = df.CustomerAge[(df.CustomerAge >= 59)]

# x = ["18-26", "27-44",
#      "45-58", "58+"]
# y = [len(age18_26.values), len(age27_44.values), len(
#     age45_58.values), len(age58above.values)]

# plt.figure(figsize=(15, 6))
# plt.bar(x, y, color=['green', 'blue', 'cyan', 'yellow'])
# plt.title("Customer and Their Ages")
# plt.xlabel("Age")
# plt.ylabel("Number of Customers")

# for i in range(len(x)):
#     plt.text(i, y[i], y[i], ha='center', va='bottom')
# plt.show()

# # 3. histogram TransactionAmount
# ta0_80 = df.TransactionAmount[(df.TransactionAmount >= 0) & (
#     df.TransactionAmount < 80)]
# ta80_210 = df.TransactionAmount[(df.TransactionAmount >= 80) & (
#     df.TransactionAmount < 210)]
# ta210_413 = df.TransactionAmount[(
#     df.TransactionAmount >= 210) & (df.TransactionAmount < 413)]
# ta413above = df.TransactionAmount[(
#     df.TransactionAmount >= 413)]

# x = ['0-80', '80-210', '210-413', '413+']
# y = [len(ta0_80.values), len(ta80_210.values),
#      len(ta210_413.values), len(ta413above.values)]

# plt.figure(figsize=(15, 6))
# plt.bar(x, y, color=['green', 'blue', 'cyan', 'yellow'])
# plt.title("Customer and Their Transaction Amount")
# plt.xlabel("Transaction Amount")
# plt.ylabel("Number of Customers")

# for i in range(len(x)):
#     plt.text(i, y[i], y[i], ha='center', va='bottom')
# plt.show()

# # 4. histogram TransactionDuration
# td10_63 = df.TransactionDuration[(df.TransactionDuration >= 10) & (
#     df.TransactionDuration < 63)]
# td63_112 = df.TransactionDuration[(df.TransactionDuration >= 63) & (
#     df.TransactionDuration < 112)]
# td112_161 = df.TransactionDuration[(
#     df.TransactionDuration >= 112) & (df.TransactionDuration < 161)]
# td161above = df.TransactionDuration[(
#     df.TransactionDuration >= 161)]

# x = ['10-63', '63-112', '112-161', '161+']
# y = [len(td10_63.values), len(td63_112.values),
#      len(td112_161.values), len(td161above.values)]

# plt.figure(figsize=(15, 6))
# plt.bar(x, y, color=['green', 'blue', 'cyan', 'yellow'])
# plt.title("Customer and Their Transaction Duration")
# plt.xlabel("Transaction Duration")
# plt.ylabel("Number of Customers")

# for i in range(len(x)):
#     plt.text(i, y[i], y[i], ha='center', va='bottom')
# plt.show()

# # 5. histogram AccountBalance
# ab101_1504 = df.AccountBalance[(df.AccountBalance >= 101) & (
#     df.AccountBalance < 1504)]
# ab1504_4734 = df.AccountBalance[(df.AccountBalance >= 1504) & (
#     df.AccountBalance < 4734)]
# ab4734_7672 = df.AccountBalance[(
#     df.AccountBalance >= 4734) & (df.AccountBalance < 7672)]
# ab7272above = df.AccountBalance[(
#     df.AccountBalance >= 7672)]

# x = ['101-1504', '1504-4734', '4734-7672', '7272+']
# y = [len(ab101_1504.values), len(ab1504_4734.values),
#      len(ab4734_7672.values), len(ab7272above.values)]

# plt.figure(figsize=(15, 6))
# plt.bar(x, y, color=['green', 'blue', 'cyan', 'yellow'])
# plt.title("Customer and Their Account Balance")
# plt.xlabel("Account Balance")
# plt.ylabel("Number of Customers")

# for i in range(len(x)):
#     plt.text(i, y[i], y[i], ha='center', va='bottom')
# plt.show()

# Visualisasi yang lebih informatif (Opsional Advanced 1)
""""""""""""""""""""""############################"""

'''PEMBERSIHAN DAN PRA PEMROSESAN DATA'''
# Mengecek dataset menggunakan isnull().sum()
# print('DATA NULL', df.isnull().sum())

# Mengecek dataset menggunakan duplicated().sum()
# print('Data duplikat:', df.duplicated().sum())

# Melakukan feature scaling menggunakan MinMaxScaler() atau StandardScalar() untuk fitur numerik.
scaler = MinMaxScaler()
numeric_columns = df.select_dtypes(include=['float64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
# Pastikan kamu menggunakan function head setelah melalukan scaling.
# print(df[numeric_columns].head())

# Melakukan drop pada kolom yang memiliki keterangan id dan IP Address
X = df.drop(columns=['TransactionID', 'AccountID',
            'DeviceID', 'MerchantID', 'IP Address', 'TransactionDate', 'PreviousTransactionDate'])

# Melakukan feature encoding menggunakan LabelEncoder() untuk fitur kategorikal.
label_encoder = LabelEncoder()
categorical_columns = ['TransactionType',
                       'Channel', 'CustomerOccupation', 'Location']
for column in categorical_columns:
    X[column] = label_encoder.fit_transform(X[column])

# Pastikan kamu menggunakan function head setelah melalukan encoding.
# print(X.head())

# Last checking gunakan columns.tolist() untuk checking seluruh fitur yang ada.
# Perbaiki kode di bawah ini tanpa menambahkan atau mengurangi cell code ini.
# print(X.columns.tolist())

# Menangani data yang hilang (bisa menggunakan dropna() atau metode imputasi fillna()).
missing_values = X.isnull().sum()
# print(missing_values[missing_values > 0])

less = missing_values[missing_values < 1880].index
over = missing_values[missing_values >= 1880].index

# print('Less: ', less)
# print('Over: ', over)

numeric_features = X[less].select_dtypes(include=['number']).columns
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
# print(X[numeric_features])

# Menghapus data duplikat menggunakan drop_duplicates().
duplicated_values = X.duplicated()
# print('\nDATA DUPLIKAT\n', X[duplicated_values])
X = X.drop_duplicates()

# Melakukan Handling Outlier Data berdasarkan jumlah outlier, apakah menggunakan metode drop atau mengisi nilai tersebut.
# for feature in X.columns:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=X[feature])
#     plt.title(f'Box Plot of {feature}')
#     plt.show()

numeric_columns = X.select_dtypes(include=['number']).columns

print("Sebelum:", len(X))
for col in numeric_columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = X[(X[col] < lower_bound) |
                 (X[col] > upper_bound)]

    X = X.drop(outliers.index)
print("Sesudah:", len(X))

print(X.head(100))
# for feature in X.columns:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=X[feature])
#     plt.title(f'Box Plot of {feature}')
#     plt.show()

# Melakukan binning data berdasarkan kondisi rentang nilai pada fitur numerik,
print('----------DATAFRAME RAW-------------')
df_raw = df
# menangani missing value pada df_raw
df_raw_missing_values = df_raw.isnull().sum()

less = missing_values[missing_values < 1880].index
over = missing_values[missing_values >= 1880].index

numeric_features = df_raw[less].select_dtypes(include=['number']).columns
df_raw[numeric_features] = df_raw[numeric_features].fillna(
    df_raw[numeric_features].median())

# menghapus duplikasi pada df_raw

print("Sebelum raw:", len(df_raw))
duplicated_values = df_raw.duplicated()
df_raw = df_raw.drop_duplicates()
print("Sesudah raw:", len(df_raw))

# lakukan pada satu sampai dua fitur numerik.
# Silahkan lakukan encode hasil binning tersebut menggunakan LabelEncoder.
# Pastikan kamu mengerjakan tahapan ini pada satu cell.
