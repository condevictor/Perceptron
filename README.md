# Perceptron

Implementação de um perceptron treinado com o algoritmo PLA para classificar dígitos do dataset `digits` entre as classes `1` e `5`.

O projeto foi feito para carregar os dados, filtrar apenas essas duas classes, adicionar bias, treinar o modelo, medir erro/acurácia em treino e teste e visualizar a fronteira de decisão em 2D com PCA.

## Visão geral

O arquivo principal é `perceptron.py`. Ele faz o seguinte:

1. Carrega os arquivos `dataset/digits.train` e `dataset/digits.test`.
2. Filtra somente as amostras das classes `1` e `5`.
3. Converte os rótulos para `+1` e `-1`.
4. Adiciona uma coluna de bias nas entradas.
5. Treina um perceptron com inicialização aleatória dos pesos e limite de épocas.
6. Calcula erro e acurácia dentro e fora da amostra.
7. Reduz os dados para 2 dimensões com PCA apenas para visualização.
8. Plota e salva a fronteira de decisão em `fronteira_decisao.png`.

## Requisitos

- Python 3.10 ou superior
- Dependências listadas em `requirements.txt`

As bibliotecas usadas no código são:

- `numpy`
- `matplotlib`
- `scikit-learn`

## Instalação

Crie e ative um ambiente virtual, se quiser isolar a execução, e instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## Como rodar

Execute o script principal a partir da raiz do projeto:

```bash
python perceptron.py
```

## O que acontece na execução

Durante a execução, o script:

- lê os dados de treino e teste;
- mantém apenas as classes `1` e `5`;
- transforma a classe `5` em `-1` e a classe `1` em `+1`;
- adiciona bias nas features;
- treina o perceptron com `max_epocas=1000` e semente `10`;
- imprime o erro e a acurácia em treino e teste;
- abre um gráfico com a fronteira de decisão;
- salva a figura como `fronteira_decisao.png`.

## Estrutura do código

### `carregar_dados(path)`
Carrega o dataset no formato texto e separa rótulos e atributos.

### `preprocessar(x, y, class_pos=1, class_neg=5)`
Filtra apenas as duas classes desejadas, converte os rótulos para `+1/-1` e adiciona bias.

### `perceptron_pla(x, y, max_epocas=1000, semente=None)`
Treina o perceptron pelo PLA. A cada época, percorre as amostras e atualiza os pesos sempre que há erro de classificação.

### `erro(y_verdadeiro, y_predito)`
Calcula a taxa de erro como a proporção de previsões incorretas.

### `plotar_fronteira(X_train, y_train, X_test, y_test, pesos)`
Reduz os dados para 2 dimensões com PCA, projeta os pesos no mesmo espaço e desenha a fronteira de decisão.

## Saída esperada

Ao final, o script imprime algo no formato:

```text
Erro dentro da amostra: 0.XXXX  |  Acurácia: 0.XXXX
Erro fora da amostra:    0.XXXX  |  Acurácia: 0.XXXX
```

Depois disso, o gráfico é exibido em uma janela e salvo no diretório atual.

## Observações

- O treinamento usa apenas as classes `1` e `5`.
- A visualização em PCA serve só para inspeção do resultado, não altera o treino.
- Se o arquivo `fronteira_decisao.png` já existir, ele será sobrescrito.

## Licença

Não definida neste repositório.
