import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import pickle


# Baca Data
df_train = pd.read_excel('Kelulusan.xlsx', sheet_name='train')
df_val = pd.read_excel('Kelulusan.xlsx', sheet_name='test')
print(df_train.dtypes)

df_train= df_train.drop(columns=['Nama', 'Jenis Kelamin', 'Prodi', 'Keselarasan Horizontal', 'Keselarasan Vertikal'])
df_val= df_val.drop(columns=['Nama', 'Jenis Kelamin', 'Prodi', 'Keselarasan Horizontal', 'Keselarasan Vertikal'])

print("Data Train:\n",df_train)
print("Tipe Data Pada Train:\n",df_train.dtypes)
print("Data validasi:\n",df_val)
print("Tipe Data Pada Vaslidasi:\n",df_val.dtypes)


#------------------------------------------------ Label Encoder dan Normalisasi Training-------------------------------------
# Label Encoder
le_IPK = LabelEncoder()
df_train['IPK'] = le_IPK.fit_transform(df_train['IPK'].astype(str))
with open('WebFlask/templates/le_IPK.pkl', 'wb') as file:
    pickle.dump(le_IPK, file)

le_Organisasi = LabelEncoder()
df_train['Organisasi'] = le_Organisasi.fit_transform(df_train['Organisasi'].astype(str))
with open('WebFlask/templates/le_Organisasi.pkl', 'wb') as file:
    pickle.dump(le_Organisasi, file)

le_Magang = LabelEncoder()
df_train['Magang MBKM'] = le_Magang.fit_transform(df_train['Magang MBKM'].astype(str))
with open('WebFlask/templates/le_Magang.pkl', 'wb') as file:
    pickle.dump(le_Magang, file)

le_Studi = LabelEncoder()
df_train['Lama Studi'] = le_Studi.fit_transform(df_train['Lama Studi'].astype(str))
with open('WebFlask/templates/le_Studi.pkl', 'wb') as file:
    pickle.dump(le_Studi, file)

le_Sertif = LabelEncoder()
df_train['Sertifikasi Tambahan'] = le_Sertif.fit_transform(df_train['Sertifikasi Tambahan'].astype(str))
with open('WebFlask/templates/le_Sertif.pkl', 'wb') as file:
    pickle.dump(le_Sertif, file)

le_bahasa= LabelEncoder()
df_train['Kemampuan Bahasa Asing'] = le_bahasa.fit_transform(df_train['Kemampuan Bahasa Asing'].astype(str))
with open('WebFlask/templates/le_bahasa.pkl', 'wb') as file:
    pickle.dump(le_bahasa, file)

le_StatusKerja= LabelEncoder()
df_train['Status Kerja'] = le_StatusKerja.fit_transform(df_train['Status Kerja'].astype(str))
with open('WebFlask/templates/le_StatusKerja.pkl', 'wb') as file:
    pickle.dump(le_StatusKerja, file)

#---------Normalisasi(Minmax)-----------]
MM_IPK = MinMaxScaler()
df_train['IPK'] = MM_IPK.fit_transform(df_train[['IPK']])
with open('WebFlask/templates/MM_IPK.pkl', 'wb') as file:
    pickle.dump(MM_IPK, file)

MM_Studi = MinMaxScaler()
df_train['Lama Studi'] = MM_Studi.fit_transform(df_train[['Lama Studi']])
with open('WebFlask/templates/MM_Studi.pkl', 'wb') as file:
    pickle.dump(MM_Studi, file)

print("Data Train Setelah Label Encoder:\n",df_train)
print("Tipe Data Pada Train Setelah Label Encoder:\n",df_train.dtypes)


#-----------------------------------Label Encoder dan MinMax Untuk Validation----------------------------
with open('WebFlask/templates/le_IPK.pkl', 'rb') as file:
    le_train_IPK = pickle.load(file)
df_val['IPK'] = le_train_IPK.transform(df_val['IPK'].astype(str))

with open('WebFlask/templates/le_Organisasi.pkl', 'rb') as file:
    le_train_Organisasi = pickle.load(file)
df_val['Organisasi'] = le_train_Organisasi.transform(df_val['Organisasi'].astype(str))

with open('WebFlask/templates/le_Magang.pkl', 'rb') as file:
    le_train_Magang = pickle.load(file)
df_val['Magang MBKM'] = le_train_Magang.transform(df_val['Magang MBKM'].astype(str))

with open('WebFlask/templates/le_Studi.pkl', 'rb') as file:
    le_train_Studi = pickle.load(file)
df_val['Lama Studi'] = le_train_Studi.transform(df_val['Lama Studi'].astype(str))

with open('WebFlask/templates/le_Sertif.pkl', 'rb') as file:
    le_train_Sertif = pickle.load(file)
df_val['Sertifikasi Tambahan'] = le_train_Sertif.transform(df_val['Sertifikasi Tambahan'].astype(str))

with open('WebFlask/templates/le_bahasa.pkl', 'rb') as file:
    le_train_bahasa = pickle.load(file)
df_val['Kemampuan Bahasa Asing'] = le_train_bahasa.transform(df_val['Kemampuan Bahasa Asing'].astype(str))

with open('WebFlask/templates/le_StatusKerja.pkl', 'rb') as file:
    le_train_StatusKerja = pickle.load(file)
df_val['Status Kerja'] = le_train_StatusKerja.transform(df_val['Status Kerja'].astype(str))

#-----------------Normalisasi-----------------
with open('WebFlask/templates/MM_IPK.pkl', 'rb') as file:
    MM_train_IPK = pickle.load(file)
df_val['IPK'] = MM_train_IPK.transform(df_val[['IPK']])

with open('WebFlask/templates/MM_Studi.pkl', 'rb') as file:
    MM_train_Studi = pickle.load(file)
df_val['Lama Studi'] = MM_train_Studi.transform(df_val[['Lama Studi']])

print("Data validasi:\n",df_val)
print("Tipe Data Pada Vaslidasi:\n",df_val.dtypes)

# Neural Netrwork
x_train = df_train.drop(columns=['Status Kerja']).values
y_train = df_train['Status Kerja'].values

x_val = df_val.drop(columns=['Status Kerja']).values
y_val = df_val['Status Kerja'].values

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return  1 / (1 + np.exp(-x))


# Fungsi untuk forward propagation
def forward(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    # Forward pass ke hidden layer
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input) # Aktivasi hidden layer

    # Forward pass ke output layer
    output_input = np.dot(hidden_output,weights_hidden_output) + bias_output
    output = sigmoid(output_input)
    return output


train_data = x_train
train_labels = y_train

# Inisialisasi bobot dan bias secara acak
np.random.seed(47)
jumlah_input = 6
jumlah_node_hidden = 10

weights_input_hidden = np.random.randn(jumlah_input, jumlah_node_hidden)
bias_hidden = np.random.randn(jumlah_node_hidden)
weights_hidden_output = np.random.randn(jumlah_node_hidden)
bias_output = np.random.randn()

# Hiperparameter
learning_rate = 0.1
epochs = 10000


# Proses Pelatihan (Training)
for epoch in range(epochs):
    for x, y in zip(train_data, train_labels):
        # Forward pass
        hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        output = sigmoid(output_input)

        # Hitung Error
        error = output - y

        # Backward pass (perbarui bobot)
        d_output = error * output * (1 - output)
        d_hidden = d_output * weights_hidden_output * hidden_output * (1 - hidden_output)

        # Update bobot dan bias
        weights_hidden_output -= learning_rate * d_output * hidden_output
        bias_output -= learning_rate * d_output
        weights_input_hidden -= learning_rate * np.outer(x, d_hidden)
        bias_hidden -= learning_rate * d_hidden

# Tes Sederhana training
test_data = x_train
output = forward(test_data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

print(" ")
for idx, val in enumerate(output):
    print(f"Prediksi {idx + 1}: {'Mendapatkan Pekerjaan Kurang dari 3 bulan' if val >= 0.5 else 'Baru Mendapatkan Pekerjaan setelah 3 bulan'}")

y_pred_training = np.where(output > 0.5, 1,0);

akurasi_training =accuracy_score(y_pred_training,y_train)
print(akurasi_training)
print(classification_report(y_train,y_pred_training))
print(confusion_matrix(y_train,y_pred_training))



# Tes Sederhana Validation
test_data = x_val
output = forward(test_data, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

print(" ")
for idx, val in enumerate(output):
    print(f"Prediksi {idx + 1}: {'Mendapatkan Pekerjaan Kurang dari 3 bulan' if val >= 0.5 else 'Baru Mendapatkan Pekerjaan setelah 3 bulan'}")

y_pred_val = np.where(output > 0.5, 1,0);


akurasi_val =accuracy_score(y_pred_val,y_val)
print(akurasi_val)
print(classification_report(y_val,y_pred_val))
print(confusion_matrix(y_val,y_pred_val))

with open('weights_input_hidden.pkl','wb') as file:
    pickle.dump(weights_input_hidden,file)

with open('bias_hidden.pkl','wb') as file:
    pickle.dump(bias_hidden,file)

with open('weights_hidden_output.pkl','wb') as file:
    pickle.dump(weights_hidden_output,file)

with open('bias_output.pkl','wb') as file:
    pickle.dump(bias_output,file)