import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Função para gerar e treinar o modelo (atualize de acordo com o código existente)
def train_model():
    time = np.arange(0, 1000, 1)
    temperature = np.random.normal(70, 5, 1000) + np.random.normal(0, 2, 1000)
    vibration = np.random.normal(30, 10, 1000) + np.random.normal(0, 2, 1000)
    failures = np.where((temperature > 80) | (vibration > 50), 1, 0)
    
    data = pd.DataFrame({
        'Time (s)': time,
        'Temperature': temperature,
        'Vibration': vibration,
        'Failure': failures
    })
    
    X = data[['Temperature', 'Vibration']]
    y = data['Failure']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Acurácia média com Cross-Validation: {np.mean(scores):.2f}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisão no conjunto de teste: {accuracy:.2f}')
    
    return data, y_pred, X_test

# Função para calcular estatísticas e exibir no painel
def calcular_estatisticas(data):
    temp_max = data['Temperature'].max()
    temp_min = data['Temperature'].min()
    temp_mean = data['Temperature'].mean()

    vib_max = data['Vibration'].max()
    vib_min = data['Vibration'].min()
    vib_mean = data['Vibration'].mean()

    num_falhas = data['Failure'].sum()

    stats = f"""Estatísticas:
    -------------------------------
    Temperatura Máxima: {temp_max:.2f}
    Temperatura Mínima: {temp_min:.2f}
    Temperatura Média: {temp_mean:.2f}
    
    Vibração Máxima: {vib_max:.2f}
    Vibração Mínima: {vib_min:.2f}
    Vibração Média: {vib_mean:.2f}
    
    Total de Falhas: {num_falhas}
    -------------------------------
    """
    return stats

# Função para gerar o relatório de falhas
def gerar_relatorio(data):
    falhas = data[data['Failure'] == 1]
    if len(falhas) == 0:
        return "Nenhuma falha foi detectada."

    relatorio = "Relatório de Falhas:\n"
    relatorio += "-" * 30 + "\n"
    for idx, row in falhas.iterrows():
        relatorio += f"Momento da Falha: {row['Time (s)']} segundos\n"
        relatorio += f"  - Temperatura: {row['Temperature']:.2f} (acima do limite)\n" if row['Temperature'] > 80 else f"  - Temperatura: {row['Temperature']:.2f} (ok)\n"
        relatorio += f"  - Vibração: {row['Vibration']:.2f} (acima do limite)\n" if row['Vibration'] > 50 else f"  - Vibração: {row['Vibration']:.2f} (ok)\n"
        relatorio += "  - Ação recomendada: "
        if row['Temperature'] > 80 and row['Vibration'] > 50:
            relatorio += "Manutenção imediata e ajuste dos equipamentos.\n"
        elif row['Temperature'] > 80:
            relatorio += "Monitorar a temperatura e revisar o sistema de refrigeração.\n"
        elif row['Vibration'] > 50:
            relatorio += "Verificar a estabilidade da máquina e ajustar o balanceamento.\n"
        relatorio += "-" * 30 + "\n"

    return relatorio

# Função para atualizar o gráfico e exibir dados
def update_dashboard():
    data, y_pred, X_test = train_model()

    ax.clear()
    ax.plot(data['Time (s)'], data['Temperature'], label='Temperatura', color='r')
    ax.plot(data['Time (s)'], data['Vibration'], label='Vibração', color='b')

    failures_temp_idx = np.where((data['Failure'] == 1) & (data['Temperature'] > 80))[0]
    ax.scatter(data['Time (s)'][failures_temp_idx], data['Temperature'][failures_temp_idx], color='black', marker='x', s=100, label='Falha (Temperatura)')

    failures_vib_idx = np.where((data['Failure'] == 1) & (data['Vibration'] > 50))[0]
    ax.scatter(data['Time (s)'][failures_vib_idx], data['Vibration'][failures_vib_idx], color='orange', marker='x', s=100, label='Falha (Vibração)')

    ax.set_title('Temperatura e Vibração ao Longo do Tempo (segundos)')
    ax.set_xlabel('Tempo (segundos)')
    ax.set_ylabel('Valor')
    ax.legend()
    canvas.draw()

    if 1 in y_pred:
        label_result.config(text="Falha prevista!", fg='red')
    else:
        label_result.config(text="Nenhuma falha prevista", fg='green')

    relatorio = gerar_relatorio(data)
    text_relatorio.delete(1.0, tk.END)
    text_relatorio.insert(tk.END, relatorio)

    stats = calcular_estatisticas(data)
    text_estatisticas.delete(1.0, tk.END)
    text_estatisticas.insert(tk.END, stats)

# Criar a janela Tkinter
window = tk.Tk()
window.title("PredictiveGuard - Monitoramento Preditivo")
window.geometry("900x600")

window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)

frame_grafico = ttk.Frame(window)
frame_grafico.grid(row=0, column=0, sticky='nsew')
frame_grafico.columnconfigure(0, weight=1)
frame_grafico.rowconfigure(0, weight=1)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
canvas.get_tk_widget().pack(fill='both', expand=True)

frame_info = ttk.Frame(window)
frame_info.grid(row=1, column=0, sticky='nsew')
frame_info.columnconfigure(0, weight=1)

btn_update = ttk.Button(frame_info, text="Atualizar Dashboard", command=update_dashboard)
btn_update.pack(pady=5, fill='x')

text_estatisticas = tk.Text(frame_info, height=5)
text_estatisticas.pack(padx=10, pady=5, fill='both', expand=True)

text_relatorio = tk.Text(frame_info, height=10)
text_relatorio.pack(padx=10, pady=5, fill='both', expand=True)

label_result = ttk.Label(frame_info, text="Previsão de Falhas", font=("Helvetica", 16))
label_result.pack(pady=5)

window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=3)
window.rowconfigure(1, weight=1)

window.mainloop()
