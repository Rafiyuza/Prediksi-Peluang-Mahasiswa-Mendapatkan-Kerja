from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask("DataMining", static_folder='Statics', static_url_path='/asset')
print(__name__)

@app.route("/")
def hal_utama():
    return render_template("Utama.html")

@app.route("/yes")
def yes():
    return render_template("Kerja.html")

@app.route("/no")
def no():
    return render_template("BelumKerja.html")

@app.route("/proses_data", methods=['POST'])
def proses_data():
    print(request.form)

    IPK = request.form['IPK']
    with open('le_IPK.pkl', 'rb') as file:
        le_IPK = pickle.load(file)
    IPK = le_IPK.transform([IPK])[0][0]
    print(IPK)

    Organisasi = request.form['Organisasi']
    with open('le_Organisasi.pkl', 'rb') as file:
        le_Organisasi = pickle.load(file)
    Organisasi = le_Organisasi.transform([[Organisasi]])[0]
    print(Organisasi)

    Magang_MBKM = request.form['Magang_MBKM']
    with open('le_Magang.pkl', 'rb') as file:
        le_Magang = pickle.load(file)
    Magang_MBKM = le_Magang.transform([[Magang_MBKM]])[0]
    print(Magang_MBKM)

    Lama_Studi = request.form['Lama_Studi']
    with open('le_Studi.pkl', 'rb') as file:
        le_Studi = pickle.load(file)
    Lama_Studi = le_Studi.transform([[Lama_Studi]])[0][0]
    print(Lama_Studi)

    Sertifikasi_Tambahan = request.form['Sertifikasi_Tambahan']
    with open('le_Sertif.pkl', 'rb') as file:
        le_Sertif = pickle.load(file)
    Sertifikasi_Tambahan = le_Sertif.transform([[Sertifikasi_Tambahan]])[0]
    print(Sertifikasi_Tambahan)

    Kemampuan_Bahasa_Asing = request.form['Kemampuan_Bahasa_Asing']
    with open('le_bahasa.pkl', 'rb') as file:
        le_bahasa = pickle.load(file)
    Kemampuan_Bahasa_Asing = le_bahasa.transform([[Kemampuan_Bahasa_Asing]])[0]
    print(Kemampuan_Bahasa_Asing)

    IPK = request.form['IPK']
    with open('MM_IPK.pkl', 'rb') as file:
        MM_IPK = pickle.load(file)
    umur = MM_IPK.transform([[IPK]])[0][0]
    print(IPK)

    Lama_Studi = request.form['Lama_Studi']
    with open('MM_Studi.pkl', 'rb') as file:
        MM_Studi = pickle.load(file)
    umur = MM_Studi.transform([[Lama_Studi]])[0][0]
    print(Lama_Studi)

    datatest = [IPK, Organisasi, Magang_MBKM, Lama_Studi, Sertifikasi_Tambahan, Kemampuan_Bahasa_Asing]

    # Load the weights and biases from the pickle files
    with open('weights_input_hidden.pkl', 'rb') as file:
        weights_input_hidden = pickle.load(file)

    with open('bias_hidden.pkl', 'rb') as file:
        bias_hidden = pickle.load(file)

    with open('weights_hidden_output.pkl', 'rb') as file:
        weights_hidden_output = pickle.load(file)

    with open('bias_output.pkl', 'rb') as file:
        bias_output = pickle.load(file)

    # Get the output from the neural network
    output = forward(datatest, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
    print(f"Model output: {output}")

    # Check the output and route accordingly
    if output > 0.5:
        return render_template("Kerja.html")
    else:
        return render_template("BelumKerja.html")


# Activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Forward propagation function
def forward(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    # Forward pass to the hidden layer
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)  # Hidden layer activation

    # Forward pass to the output layer
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = sigmoid(output_input)
    return output


if __name__ == "__main__":
    app.run()