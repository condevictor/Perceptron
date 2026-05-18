import numpy as np 

def carregar_dados(path):
    dados = np.loadtxt(path)
    y = dados[:, 0]
    x = dados[:, 1:]
    return x, y

def preprocessar(x, y, class_pos = 1, class_neg = -1):
    mascara = ((y == class_pos) | (y == class_neg))
    x = x[mascara]
    y = y[mascara]

    y = np.where(y == class_neg, -1, 1)
    #adicionando bias
    bias = np.ones(x.shape[0])

    x = np.c_[bias, x]

    return x, y

if __name__ == "__main__":
    x_train, y_train = carregar_dados("dataset/digits.train")
    x_test, y_test = carregar_dados("dataset/digits.test")
    x_train, y_train = preprocessar(x_train, y_train, class_pos = 1, class_neg = 5)
    x_test, y_test = preprocessar(x_test, y_test, class_pos = 1, class_neg = 5)