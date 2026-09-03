# PredictiveGuard - Monitoramento Preditivo Industrial

O **PredictiveGuard** é um sistema avançado de manutenção preditiva com interface gráfica interativa, desenvolvido para prever falhas em maquinário industrial utilizando Machine Learning. O sistema processa dados em tempo real (streaming), aplica algoritmos de classificação estatística e persiste o histórico em banco de dados relacional para apoiar a tomada de decisão.

## 🚀 Funcionalidades Principais

* **Streaming de Dados em Tempo Real:** Geração contínua e calibração realista de sensores industriais (Temperatura, Vibração, Carga, Temperatura Ambiente e Umidade).
* **Inteligência Artificial (Random Forest):** Treinamento de modelo classificatório para prever a probabilidade de risco crítico, apontando ativamente qual o principal fator de falha na janela de observação.
* **Dashboard Moderno (CustomTkinter):** Interface gráfica no padrão Dark Mode com gráficos do `matplotlib` integrados, exibindo uma janela deslizante dos últimos 60 segundos.
* **Latching e Smoothing de Alarmes:** Retenção inteligente de alertas e suavização da probabilidade de falhas para evitar fadiga visual do operador.
* **Persistência de Dados (SQLite):** Banco de dados integrado que grava todo o histórico de monitoramento automaticamente, garantindo que nenhum dado seja perdido ao encerrar o sistema.
* **Relatórios Automatizados (PDF):** Exportação inteligente do painel de operações, anexando uma captura gráfica do momento da falha juntamente com estatísticas gerais.

## 🛠️ Tecnologias Utilizadas

* **Python 3** (Linguagem Principal)
* **CustomTkinter & Matplotlib** (Front-end e Visualização de Dados)
* **Scikit-Learn & Pandas** (Machine Learning e Engenharia de Dados)
* **SQLite3** (Persistência e Banco de Dados)
* **FPDF** (Geração de Relatórios Documentais)

## ⚙️ Como Instalar e Rodar

**1. Clone o repositório**
Navegue até a pasta desejada em seu terminal e clone o projeto.

**2. Instale as dependências**
Recomenda-se o uso de um ambiente virtual (venv). Instale as bibliotecas utilizando o comando:
```bash
pip install -r requirements.txt