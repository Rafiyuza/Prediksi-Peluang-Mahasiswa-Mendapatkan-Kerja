import pickle
import numpy as np

# Fungsi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Fungsi forward propagation
def forward(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = sigmoid(output_input)
    return output

# ==========================
# INPUT DATA MANUAL DARI TERMINAL
# ==========================

IPK_raw = input("Masukkan IPK (contoh: 3.0): ")  # IPK numerik, tidak perlu encode
Organisasi = input("Apakah aktif organisasi? (Ya/Tidak): ")
Magang_MBKM = input("Ikut magang MBKM? (Ya/Tidak): ")
Lama_Studi_raw = input("Masukkan Lama Studi (semester): ")  # Lama studi numerik
Sertifikasi_Tambahan = input("Punya sertifikasi tambahan? (Ya/Tidak): ")
Kemampuan_Bahasa_Asing = input("Bisa bahasa asing? (Ya/Tidak): ")

# ==========================
# TRANSFORMASI: ENCODING & SCALING
# ==========================

# Normalisasi IPK (MinMaxScaler)
with open('MM_IPK.pkl', 'rb') as file:
    MM_IPK = pickle.load(file)
IPK_scaled = MM_IPK.transform([[float(IPK_raw)]])[0][0]

# Organisasi (LabelEncoder)
with open('le_Organisasi.pkl', 'rb') as file:
    le_Organisasi = pickle.load(file)
Organisasi_enc = le_Organisasi.transform([[Organisasi]])[0]

# Magang MBKM (LabelEncoder)
with open('le_Magang.pkl', 'rb') as file:
    le_Magang = pickle.load(file)
Magang_enc = le_Magang.transform([[Magang_MBKM]])[0]

# Normalisasi Lama Studi
with open('MM_Studi.pkl', 'rb') as file:
    MM_Studi = pickle.load(file)
Lama_Studi_scaled = MM_Studi.transform([[float(Lama_Studi_raw)]])[0][0]

# Sertifikasi (LabelEncoder)
with open('le_Sertif.pkl', 'rb') as file:
    le_Sertif = pickle.load(file)
Sertif_enc = le_Sertif.transform([[Sertifikasi_Tambahan]])[0]

# Bahasa Asing (LabelEncoder)
with open('le_bahasa.pkl', 'rb') as file:
    le_bahasa = pickle.load(file)
Bahasa_enc = le_bahasa.transform([[Kemampuan_Bahasa_Asing]])[0]

# ==========================
# GABUNGKAN SEMUA FITUR KE ARRAY
# ==========================

x_input = [IPK_scaled, Organisasi_enc, Magang_enc, Lama_Studi_scaled, Sertif_enc, Bahasa_enc]

# ==========================
# LOAD BOBOT DAN BIAS
# ==========================

with open('weights_input_hidden.pkl', 'rb') as file:
    weights_input_hidden = pickle.load(file)

with open('bias_hidden.pkl', 'rb') as file:
    bias_hidden = pickle.load(file)

with open('weights_hidden_output.pkl', 'rb') as file:
    weights_hidden_output = pickle.load(file)

with open('bias_output.pkl', 'rb') as file:
    bias_output = pickle.load(file)

# ==========================
# PREDIKSI
# ==========================

output = forward(x_input, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

print(f"\nOutput prediksi: {output:.4f}")

if output > 0.5:
    print("✅ Prediksi: Lulus Cepat Kerja (≤ 3 bulan)")
else:
    print("❌ Prediksi: Belum Kerja setelah lulus")
