# SCIC - Sistema de Comunicação Interplanetária da Colônia

## Sobre o projeto

O SCIC é um protótipo desenvolvido para a missão fictícia Aurora Siger.

O sistema organiza e analisa informações operacionais e de comunicação da colônia, permitindo estudar latência, erros numéricos, previsão, priorização de alertas e busca eficiente de registros.

## Funcionalidades

O sistema permite:

- carregar dados da colônia em CSV;
- consultar módulos;
- calcular erro absoluto;
- calcular erro relativo;
- analisar ponto flutuante;
- executar regressão linear;
- calcular MAE, MSE, RMSE e R²;
- calcular AIC e BIC;
- executar Grid Search;
- priorizar alertas utilizando heap;
- realizar busca por prefixo utilizando trie;
- converter códigos entre decimal, binário e hexadecimal;
- calcular potência elétrica;
- aplicar a Lei de Ohm;
- executar uma simulação simples com o método de Euler;
- apresentar uma análise final da comunicação.

## Tecnologias utilizadas

- Python
- NumPy
- Pandas
- scikit-learn

## Instalação

Com o terminal aberto na pasta do projeto, instale todas as dependências listadas no arquivo `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

É importante executar o comando usando o mesmo interpretador Python selecionado no IDE.

## Execução

Mantenha os arquivos:

```txt
codigo_fonte.py
dados_aurora_siger.csv
```

na mesma pasta.

```bash
python -X utf8 codigo_fonte.py
```

## Exemplo

```txt
===== SCIC - AURORA SIGER =====

1 - Visualizar dados
2 - Consultar módulo
3 - Analisar erros numéricos
...
```

Ao selecionar a análise de erros, o sistema compara a latência prevista com a observada.

Exemplo:

```txt
Módulo: Agricultura Verde
Latência observada: 31.8 ms
Latência prevista: 27.5 ms
Erro absoluto: 4.3 ms
```

## Heap

O heap é utilizado como fila de prioridade.

Alertas de módulos importantes e com maior diferença de latência recebem maior prioridade operacional.

## Trie

A trie permite busca por prefixo.

Por exemplo:

```txt
Prefixo: com
```

pode retornar:

```txt
comunicação central
comunicação secundária
```

## Modelo preditivo

Foi utilizado um modelo de regressão linear para estimar latência.

As métricas utilizadas são:

- MAE
- MSE
- RMSE
- R²

Também são demonstrados AIC, BIC e Grid Search.

## Eletricidade

O sistema relaciona comunicação a conceitos de arquitetura computacional e eletricidade.

A potência é calculada por:

```txt
P = V × I
```

A resistência pode ser calculada pela Lei de Ohm:

```txt
R = V / I
```

## Estrutura

```txt
SCIC_Aurora_Siger/
├── codigo_fonte.py
├── dados_aurora_siger.csv
├── requirements.txt
├── relatorio_tecnico.pdf
├── README.md
└── link_video.txt
```

## Observação

O projeto utiliza dados simulados para fins acadêmicos. Não representa uma rede real de comunicação espacial.
