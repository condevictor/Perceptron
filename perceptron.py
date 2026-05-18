import numpy as np 

def carregar_dados(path):
    dados = np.loadtxt(path)
    y = dados[:, 0]
    x = dados[:, 1:]
    return x, y

def preprocessar(x, y, class_pos = 1, class_neg = 5):
    mascara = ((y == class_pos) | (y == class_neg))
    x = x[mascara]
    y = y[mascara]

    y = np.where(y == class_neg, -1, 1)
    #adicionando bias
    bias = np.ones(x.shape[0])

    x = np.c_[bias, x]

    return x, y

def perceptron_pla(x, y, max_epocas=1000, semente=None):
    linhas = x.shape[0]
    colunas_atributos = x.shape[1]
    if semente is not None:
        np.random.seed(semente)
    w = np.random.rand(colunas_atributos) * 0.01
    for epoca in range(max_epocas):
        erros = 0
        for i in range(linhas):
            if np.sign(np.dot(linhas[i], w)) != y[i]:
                w = w + (y[i] * x[i])
                erros += 1
        if erros == 0:
            print(f"PLA convergiu na época {epoca + 1}")
            break
    else:
        print(f"Parou após {max_epocas} épocas. Erros na última época: {erros}")
        return w


if __name__ == "__main__":
    x_train, y_train = carregar_dados("dataset/digits.train")
    x_test, y_test = carregar_dados("dataset/digits.test")
    x_train, y_train = preprocessar(x_train, y_train, class_pos = 1, class_neg = 5)
    x_test, y_test = preprocessar(x_test, y_test, class_pos = 1, class_neg = 5)

    w = perceptron_pla(x_train, y_train, max_epocas=1000, semente=87)