import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
            if np.sign(np.dot(x[i], w)) != y[i]:
                w = w + (y[i] * x[i])
                erros += 1
        if erros == 0:
            print(f"PLA convergiu na época {epoca + 1}")
            break
    else:
        print(f"Parou após {max_epocas} épocas. Erros na última época: {erros}")
    return w
    
def erro(y_verdadeiro, y_predito):
    return np.mean(y_predito != y_verdadeiro)

def plotar_fronteira(X_train, y_train, X_test, y_test, pesos):
    pca = PCA(n_components=2)
    X_train_sem_bias = X_train[:, 1:]
    X_test_sem_bias = X_test[:, 1:]
    # Fit PCA apenas nos treino
    X_train_2d = pca.fit_transform(X_train_sem_bias)
    X_test_2d = pca.transform(X_test_sem_bias)

    # Extrair média e componentes
    media = pca.mean_   # vetor média das features (sem bias)
    componentes = pca.components_   # shape (2, n_features)

    w0 = pesos[0]
    w_feat = pesos[1:]

    # Termo de bias ajustado: w0' = w0 + w_feat · media
    w0_ajustado = w0 + np.dot(w_feat, media)

    # Coeficientes angulares no espaço PCA
    w_pca = componentes @ w_feat   # shape (2,)

    # Agora a reta no espaço PCA: w0_ajustado + w_pca[0]*z1 + w_pca[1]*z2 = 0
    z1_vals = np.linspace(X_train_2d[:, 0].min(), X_train_2d[:, 0].max(), 100)
    if abs(w_pca[1]) > 1e-6:
        z2_vals = -(w0_ajustado + w_pca[0] * z1_vals) / w_pca[1]
    else:
        # Reta vertical
        z1_vals = np.full(100, -w0_ajustado / w_pca[0])
        z2_vals = np.linspace(X_train_2d[:, 1].min(), X_train_2d[:, 1].max(), 100)

    # Plot (igual ao seu, apenas com a reta corrigida)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_2d[y_train==1, 0], X_train_2d[y_train==1, 1], c='blue', marker='o', alpha=0.6, label='Treino +1')
    plt.scatter(X_train_2d[y_train==-1, 0], X_train_2d[y_train==-1, 1], c='red', marker='x', alpha=0.6, label='Treino -1')
    plt.scatter(X_test_2d[y_test==1, 0], X_test_2d[y_test==1, 1], c='cyan', marker='s', alpha=0.3, label='Teste +1')
    plt.scatter(X_test_2d[y_test==-1, 0], X_test_2d[y_test==-1, 1], c='magenta', marker='s', alpha=0.3, label='Teste -1')
    plt.plot(z1_vals, z2_vals, 'k-', linewidth=2, label='Fronteira')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Perceptron – Fronteira de Decisão (PCA)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fronteira_decisao.png')
    plt.show()


if __name__ == "__main__":
    x_train, y_train = carregar_dados("dataset/digits.train")
    x_test, y_test = carregar_dados("dataset/digits.test")
    x_train, y_train = preprocessar(x_train, y_train, class_pos = 1, class_neg = 5)
    x_test, y_test = preprocessar(x_test, y_test, class_pos = 1, class_neg = 5)

    w = perceptron_pla(x_train, y_train, max_epocas=1, semente=87)

    pred_train = np.sign(np.dot(x_train, w))
    pred_test  = np.sign(np.dot(x_test, w))
    
    erro_train = erro(y_train, pred_train)
    erro_test = erro(y_test, pred_test)
    ac_train = 1 - erro_train
    ac_test = 1 - erro_test
    
    print(f"Erro dentro da amostra: {erro_train:.4f}  |  Acurácia: {ac_train:.4f}")
    print(f"Erro fora da amostra:    {erro_test:.4f}  |  Acurácia: {ac_test:.4f}")

    plotar_fronteira(x_train, y_train, x_test, y_test, w)