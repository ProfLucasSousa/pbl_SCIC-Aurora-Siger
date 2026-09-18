# ============================================================
# SCIC - Sistema de Comunicação Interplanetária da Colônia
# Missão Aurora Siger
# ============================================================

import heapq
import math

try:
    import numpy as np
    import pandas as pd

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import GridSearchCV, train_test_split
except ImportError as erro:
    pacote = getattr(erro, "name", "uma dependência")
    raise SystemExit(
        f"Dependência não instalada: {pacote}.\n"
        "Instale as dependências com:\n"
        "  python -m pip install -r requirements.txt"
    ) from erro

ARQUIVO_DADOS = "dados_aurora_siger.csv"


# ============================================================
# 1. CARREGAMENTO E ORGANIZAÇÃO DOS DADOS
# ============================================================

def carregar_dados():
    try:
        dados = pd.read_csv(ARQUIVO_DADOS)
        print("\nBase de dados carregada com sucesso.")
        return dados
    except FileNotFoundError:
        print(f"\nERRO: arquivo {ARQUIVO_DADOS} não encontrado.")
        return None


def exibir_dados(dados):
    print("\n===== DADOS DA AURORA SIGER =====")
    print(dados.to_string(index=False))


def consultar_modulo(dados):
    nome = input("\nDigite parte do nome do módulo: ").strip().lower()

    resultado = dados[
        dados["modulo"].str.lower().str.contains(nome)
    ]

    if resultado.empty:
        print("Nenhum módulo encontrado.")
    else:
        print("\n===== RESULTADO =====")
        print(resultado.to_string(index=False))


# ============================================================
# 2. INDICADORES E ANÁLISE NUMÉRICA
# ============================================================

def calcular_erros(dados):
    dados = dados.copy()

    dados["erro_absoluto"] = abs(
        dados["latencia_observada"]
        - dados["latencia_prevista"]
    )

    dados["erro_relativo_percentual"] = (
        dados["erro_absoluto"]
        / dados["latencia_observada"]
    ) * 100

    return dados


def analisar_erros(dados):
    dados_erros = calcular_erros(dados)

    print("\n===== ERRO ABSOLUTO E RELATIVO =====")

    colunas = [
        "modulo",
        "latencia_observada",
        "latencia_prevista",
        "erro_absoluto",
        "erro_relativo_percentual"
    ]

    print(
        dados_erros[colunas]
        .round(3)
        .to_string(index=False)
    )

    media_absoluta = dados_erros["erro_absoluto"].mean()
    media_relativa = dados_erros[
        "erro_relativo_percentual"
    ].mean()

    print("\nErro absoluto médio:", round(media_absoluta, 3), "ms")
    print("Erro relativo médio:", round(media_relativa, 3), "%")

    print("\nInterpretação:")

    if media_relativa <= 5:
        print("Diferença média baixa: previsões muito próximas das observações.")
    elif media_relativa <= 10:
        print("Diferença moderada: recomenda-se monitoramento.")
    else:
        print("ALERTA: diferença elevada entre previsão e observação.")

    return dados_erros


# ============================================================
# 3. PRECISÃO NUMÉRICA E PONTO FLUTUANTE
# ============================================================

def demonstrar_ponto_flutuante():
    valor = 0.1 + 0.2

    print("\n===== PONTO FLUTUANTE =====")
    print("0.1 + 0.2 =", valor)
    print("0.1 + 0.2 == 0.3 ?", valor == 0.3)

    print(
        "\nComputadores representam muitos números reais "
        "por aproximação binária."
    )

    print(
        "Por isso, pequenas diferenças podem aparecer "
        "em cálculos numéricos."
    )


# ============================================================
# 4. MODELO DE REGRESSÃO
# ============================================================

def preparar_modelo(dados):
    # Variáveis de entrada relacionadas ao funcionamento
    X = dados[
        [
            "tensao",
            "corrente",
            "potencia_transmissao",
            "ciclo"
        ]
    ]

    y = dados["latencia_observada"]

    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )


def avaliar_regressao(dados):
    X_treino, X_teste, y_treino, y_teste = preparar_modelo(dados)

    modelo = LinearRegression()
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)

    mae = mean_absolute_error(y_teste, previsoes)
    mse = mean_squared_error(y_teste, previsoes)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_teste, previsoes)

    print("\n===== MODELO DE REGRESSÃO LINEAR =====")
    print("MAE :", round(mae, 4))
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    print("\n===== VALORES REAIS X PREVISTOS =====")

    comparacao = pd.DataFrame({
        "real": y_teste.values,
        "previsto": previsoes
    })

    print(comparacao.round(3).to_string(index=False))

    interpretar_metricas(mae, mse, rmse, r2)

    calcular_aic_bic(
        y_teste.values,
        previsoes,
        numero_parametros=X_treino.shape[1] + 1
    )

    return modelo


def interpretar_metricas(mae, mse, rmse, r2):
    print("\n===== INTERPRETAÇÃO =====")

    print(
        "MAE representa o erro médio da previsão "
        "na mesma unidade da latência."
    )

    print(
        "MSE penaliza erros grandes de forma mais intensa."
    )

    print(
        "RMSE também está na unidade original e evidencia "
        "a presença de erros maiores."
    )

    print(
        "R² indica quanto da variação dos dados é explicada "
        "pelo modelo."
    )

    if rmse > mae * 1.5:
        print(
            "\nA diferença entre RMSE e MAE sugere "
            "a existência de alguns erros mais elevados."
        )

    if r2 >= 0.8:
        print("O modelo apresenta bom ajuste para esta base simulada.")
    elif r2 >= 0.5:
        print("O ajuste é moderado.")
    else:
        print("O modelo deve ser interpretado com cautela.")


# ============================================================
# 5. AIC E BIC
# ============================================================

def calcular_aic_bic(reais, previstos, numero_parametros):
    n = len(reais)

    rss = np.sum(
        (np.array(reais) - np.array(previstos)) ** 2
    )

    if rss <= 0:
        rss = 1e-10

    aic = (
        n * np.log(rss / n)
        + 2 * numero_parametros
    )

    bic = (
        n * np.log(rss / n)
        + numero_parametros * np.log(n)
    )

    print("\n===== CRITÉRIOS DE COMPARAÇÃO =====")
    print("AIC:", round(aic, 3))
    print("BIC:", round(bic, 3))

    print(
        "AIC e BIC são mais úteis para comparar modelos: "
        "valores menores tendem a indicar melhor equilíbrio "
        "entre ajuste e complexidade."
    )


# ============================================================
# 6. GRID SEARCH
# ============================================================

def executar_grid_search(dados):
    X = dados[
        [
            "tensao",
            "corrente",
            "potencia_transmissao",
            "ciclo"
        ]
    ]

    y = dados["latencia_observada"]

    modelo = Ridge()

    parametros = {
        "alpha": [0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        modelo,
        parametros,
        cv=3,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X, y)

    print("\n===== GRID SEARCH =====")
    print("Melhor alpha:", grid.best_params_["alpha"])

    print(
        "O Grid Search testou diferentes valores de "
        "regularização e escolheu o de melhor desempenho."
    )


# ============================================================
# 7. HEAP / FILA DE PRIORIDADE
# ============================================================

def criar_heap_alertas(dados):
    heap = []

    for indice, linha in dados.iterrows():

        prioridade = int(linha["prioridade"])

        atraso = max(
            0,
            linha["latencia_observada"]
            - linha["latencia_prevista"]
        )

        # heapq é min-heap.
        # Valor negativo faz prioridades maiores aparecerem primeiro.
        pontuacao = -(prioridade * 100 + atraso)

        alerta = (
            pontuacao,
            linha["modulo"],
            linha["alerta"],
            linha["latencia_observada"]
        )

        heapq.heappush(heap, alerta)

    return heap


def exibir_alertas_priorizados(dados):
    heap = criar_heap_alertas(dados)

    print("\n===== ALERTAS PRIORIZADOS COM HEAP =====")

    posicao = 1

    while heap:
        pontuacao, modulo, alerta, latencia = heapq.heappop(heap)

        print(
            f"{posicao}. {modulo} | "
            f"{alerta} | Latência: {latencia} ms"
        )

        posicao += 1


# ============================================================
# 8. TRIE
# ============================================================

class NoTrie:
    def __init__(self):
        self.filhos = {}
        self.fim = False


class Trie:

    def __init__(self):
        self.raiz = NoTrie()

    def inserir(self, palavra):
        atual = self.raiz

        for caractere in palavra.lower():
            if caractere not in atual.filhos:
                atual.filhos[caractere] = NoTrie()

            atual = atual.filhos[caractere]

        atual.fim = True

    def _coletar(self, no, prefixo, resultados):

        if no.fim:
            resultados.append(prefixo)

        for caractere, filho in no.filhos.items():
            self._coletar(
                filho,
                prefixo + caractere,
                resultados
            )

    def buscar_prefixo(self, prefixo):
        atual = self.raiz
        prefixo = prefixo.lower()

        for caractere in prefixo:

            if caractere not in atual.filhos:
                return []

            atual = atual.filhos[caractere]

        resultados = []

        self._coletar(
            atual,
            prefixo,
            resultados
        )

        return resultados


def construir_trie(dados):
    trie = Trie()

    for nome in dados["modulo"].unique():
        trie.inserir(nome)

    for codigo in dados["codigo_sensor"].unique():
        trie.inserir(str(codigo))

    return trie


def buscar_com_trie(dados):
    trie = construir_trie(dados)

    prefixo = input(
        "\nDigite prefixo de módulo ou sensor: "
    )

    resultados = trie.buscar_prefixo(prefixo)

    print("\n===== RESULTADOS DA TRIE =====")

    if resultados:
        for resultado in resultados:
            print("-", resultado)
    else:
        print("Nenhum registro encontrado.")


# ============================================================
# 9. BASES NUMÉRICAS
# ============================================================

def converter_bases():
    try:
        numero = int(
            input("\nDigite um código decimal de sensor: ")
        )
    except ValueError:
        print("Valor inválido.")
        return

    print("\n===== CONVERSÃO DE BASES =====")
    print("Decimal     :", numero)
    print("Binário     :", bin(numero))
    print("Hexadecimal :", hex(numero))

    print(
        "\nEssas representações podem ser usadas por "
        "dispositivos e interfaces computacionais."
    )


# ============================================================
# 10. ELETRICIDADE BÁSICA
# ============================================================

def eletricidade_comunicacao():
    try:
        tensao = float(input("\nTensão do transmissor (V): "))
        corrente = float(input("Corrente do transmissor (A): "))
    except ValueError:
        print("Valores inválidos.")
        return

    potencia = tensao * corrente

    print("\n===== ELETRICIDADE DO TRANSMISSOR =====")
    print(f"Potência P = V × I = {potencia:.2f} W")

    if corrente != 0:
        resistencia = tensao / corrente

        print(
            f"Lei de Ohm: R = V / I = "
            f"{resistencia:.2f} ohms"
        )


# ============================================================
# 11. SIMULAÇÃO NUMÉRICA COM EULER
# ============================================================

def simular_latencia_euler():
    print("\n===== SIMULAÇÃO COM MÉTODO DE EULER =====")

    latencia = 20.0
    taxa_variacao = 0.8
    passo = 1.0
    ciclos = 5

    print("Modelo simplificado:")
    print("dL/dt = 0.8 ms por ciclo\n")

    for ciclo in range(ciclos + 1):
        print(
            f"Ciclo {ciclo}: "
            f"latência aproximada = {latencia:.2f} ms"
        )

        latencia = latencia + passo * taxa_variacao

    print(
        "\nO método de Euler aproxima a evolução "
        "da variável em passos discretos."
    )


# ============================================================
# 12. ANÁLISE FINAL
# ============================================================

def analise_final(dados):
    erros = calcular_erros(dados)

    media_latencia = dados["latencia_observada"].mean()

    alertas = dados[
        dados["status"].str.lower() == "alerta"
    ]

    maior_erro = erros.loc[
        erros["erro_absoluto"].idxmax()
    ]

    potencia_media = dados[
        "potencia_transmissao"
    ].mean()

    print("\n===== ANÁLISE FINAL DO SCIC =====")

    print(
        "Latência média observada:",
        round(media_latencia, 2),
        "ms"
    )

    print(
        "Quantidade de módulos/registros em alerta:",
        len(alertas)
    )

    print(
        "Maior diferença entre latência prevista e observada:",
        maior_erro["modulo"],
        "-",
        round(maior_erro["erro_absoluto"], 2),
        "ms"
    )

    print(
        "Potência média de transmissão:",
        round(potencia_media, 2),
        "W"
    )

    if len(alertas) > 0:
        print(
            "\nRecomendação: manter monitoramento contínuo "
            "e analisar os alertas de maior prioridade."
        )

    print(
        "\nO uso combinado de previsão, métricas, heap e trie "
        "permite detectar desvios, priorizar ocorrências e "
        "localizar registros rapidamente."
    )


# ============================================================
# 13. DISPOSITIVOS E GERENCIAMENTO INTELIGENTE
# ============================================================

def explicar_arquitetura():
    print("\n===== ARQUITETURA CONCEITUAL =====")

    print("""
ENTRADAS:
- Sensores de comunicação
- Medidores de tensão e corrente
- Registros de latência
- Dispositivos IoT simulados

PROCESSAMENTO:
- SCIC em Python
- Pandas / NumPy
- Modelo de regressão
- Heap
- Trie

SAÍDAS:
- Terminal
- Relatórios
- Alertas
- Métricas
- Recomendações

GERENCIAMENTO INTELIGENTE:
- Monitoramento contínuo
- Detecção de anomalias
- Priorização automática
- Redundância de enlaces
- Manutenção preditiva
- Redução de desperdício energético
""")


# ============================================================
# 14. MENU
# ============================================================

def menu():

    dados = carregar_dados()

    if dados is None:
        return

    while True:

        print("\n========================================")
        print(" SCIC - AURORA SIGER")
        print("========================================")
        print("1  - Visualizar dados")
        print("2  - Consultar módulo")
        print("3  - Analisar erros numéricos")
        print("4  - Demonstrar ponto flutuante")
        print("5  - Executar regressão e métricas")
        print("6  - Executar Grid Search")
        print("7  - Priorizar alertas com heap")
        print("8  - Buscar por prefixo com trie")
        print("9  - Converter bases numéricas")
        print("10 - Calcular potência / Lei de Ohm")
        print("11 - Simular latência com Euler")
        print("12 - Exibir arquitetura conceitual")
        print("13 - Exibir análise final")
        print("0  - Sair")

        opcao = input("\nEscolha uma opção: ")

        if opcao == "1":
            exibir_dados(dados)

        elif opcao == "2":
            consultar_modulo(dados)

        elif opcao == "3":
            analisar_erros(dados)

        elif opcao == "4":
            demonstrar_ponto_flutuante()

        elif opcao == "5":
            avaliar_regressao(dados)

        elif opcao == "6":
            executar_grid_search(dados)

        elif opcao == "7":
            exibir_alertas_priorizados(dados)

        elif opcao == "8":
            buscar_com_trie(dados)

        elif opcao == "9":
            converter_bases()

        elif opcao == "10":
            eletricidade_comunicacao()

        elif opcao == "11":
            simular_latencia_euler()

        elif opcao == "12":
            explicar_arquitetura()

        elif opcao == "13":
            analise_final(dados)

        elif opcao == "0":
            print("\nEncerrando SCIC...")
            break

        else:
            print("Opção inválida.")


if __name__ == "__main__":
    menu()
