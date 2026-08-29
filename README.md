# predictive-guard
# PredictiveGuard - Monitoramento Preditivo

O **PredictiveGuard** é um simulador de manutenção preditiva com interface gráfica interativa, desenvolvido para prever falhas em maquinário industrial utilizando Machine Learning. O sistema gera dados simulados em tempo real, aplica algoritmos de classificação e exibe dashboards visuais para auxiliar na tomada de decisão rápida.

## 🚀 Funcionalidades Principais

* **Simulação de Dados:** Geração de dados de sensores industriais, incluindo Temperatura, Vibração, Carga da Máquina, Temperatura Ambiente, Umidade e Idade da Máquina.
* **Machine Learning:** Utiliza o algoritmo `RandomForestClassifier` com validação cruzada para prever falhas com base em regras de negócio e anomalias.
* **Dashboard Interativo (Tkinter):** Interface gráfica que exibe gráficos da evolução de temperatura, vibração e carga ao longo do tempo usando `matplotlib`.
* **Relatórios e Estatísticas:** Cálculo de estatísticas gerais, detecção de pontos de falha e geração de recomendações de correção na tela.
* **Exportação em PDF:** Capacidade de gerar e salvar um relatório consolidado de manutenção em formato PDF.

## 🛠️ Tecnologias e Dependências

O projeto foi construído em Python e utiliza a biblioteca nativa `tkinter` para a interface. Para rodar a aplicação, você precisará instalar as bibliotecas externas abaixo:

* `numpy` (Manipulação de arrays e cálculos numéricos)
* `pandas` (Estruturação de dados via DataFrames)
* `scikit-learn` (Treinamento do modelo de Machine Learning)
* `matplotlib` (Geração dos gráficos para o Dashboard)
* `fpdf` (Exportação dos relatórios em PDF)

## ⚙️ Como Instalar e Rodar

**1. Clone o repositório ou baixe os arquivos**
Abra o seu terminal e navegue até a pasta onde deseja salvar o projeto.

**2. Instale as dependências necessárias**
Recomenda-se o uso de um ambiente virtual (venv), mas você pode instalar as dependências globalmente usando o `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib fpdf
```

**3. Execute a aplicação**
Com as dependências instaladas, basta executar o arquivo principal da interface gráfica no terminal:
```bash
python gui.py
```
Uma janela do CustomTkinter se abrirá exibindo o painel principal. Clique em Atualizar Dashboard para gerar os dados simulados, treinar o modelo e visualizar os gráficos e o relatório de falhas. E caso deseje, clique em Exportar Relatório para gerar o PDF de manutenção.