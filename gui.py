from fpdf import FPDF
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# Variáveis globais para os limites
limites = {
    "temp_max": 80,
    "vib_max": 50,
    "load_max": 90
}

# Função para abrir a janela de configurações
def abrir_configuracoes():
    def salvar_configuracoes():
        try:
            limites["temp_max"] = float(entry_temp.get())
            limites["vib_max"] = float(entry_vib.get())
            limites["load_max"] = float(entry_load.get())
            messagebox.showinfo("Configurações Salvas", "Os limites foram atualizados com sucesso!")
            config_window.destroy()
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")

    # Janela de configurações
    config_window = tk.Toplevel(window)
    config_window.title("Configurações de Limites")
    config_window.geometry("400x300")

    # Labels e entradas para os limites
    tk.Label(config_window, text="Temperatura Máxima (°C):", font=("Helvetica", 12)).pack(pady=5)
    entry_temp = tk.Entry(config_window, font=("Helvetica", 12))
    entry_temp.insert(0, str(limites["temp_max"]))
    entry_temp.pack(pady=5)

    tk.Label(config_window, text="Vibração Máxima:", font=("Helvetica", 12)).pack(pady=5)
    entry_vib = tk.Entry(config_window, font=("Helvetica", 12))
    entry_vib.insert(0, str(limites["vib_max"]))
    entry_vib.pack(pady=5)

    tk.Label(config_window, text="Carga Máxima (%):", font=("Helvetica", 12)).pack(pady=5)
    entry_load = tk.Entry(config_window, font=("Helvetica", 12))
    entry_load.insert(0, str(limites["load_max"]))
    entry_load.pack(pady=5)

    # Botão para salvar configurações
    btn_salvar = tk.Button(config_window, text="Salvar", font=("Helvetica", 12), bg="#007BFF", fg="white", padx=10, pady=5, command=salvar_configuracoes)
    btn_salvar.pack(pady=20)

# Função para gerar dados simulados com base nos limites definidos pelo usuário
def gerar_dados():
    time = np.arange(0, 1000, 1)  # Tempo em segundos
    temperature = np.random.normal(70, 5, 1000)  # Temperatura operacional
    vibration = np.random.normal(30, 10, 1000)  # Vibração operacional
    load = np.random.uniform(50, 100, 1000)  # Carga da máquina (%)
    ambient_temp = np.random.uniform(20, 40, 1000)  # Temperatura ambiente (°C)
    humidity = np.random.uniform(30, 80, 1000)  # Umidade ambiente (%)
    machine_age = np.random.uniform(1, 10, 1000)  # Idade da máquina (anos)

    failures = np.where(
        (temperature > limites["temp_max"]) |
        (vibration > limites["vib_max"]) |
        (load > limites["load_max"]),
        1,  # Falha ocorre
        0   # Sem falha
    )

    data = pd.DataFrame({
        'Time (s)': time,
        'Temperature': temperature,
        'Vibration': vibration,
        'Load (%)': load,
        'Ambient Temp (°C)': ambient_temp,
        'Humidity (%)': humidity,
        'Machine Age (years)': machine_age,
        'Failure': failures
    })
    return data

# Função para treinar o modelo
def treinar_modelo(data):
    X = data[['Temperature', 'Vibration', 'Load (%)', 'Ambient Temp (°C)', 'Humidity (%)', 'Machine Age (years)']]
    y = data['Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisão do modelo: {accuracy:.2f}")
    return model

# Função para calcular estatísticas gerais e recomendações
def calcular_estatisticas(data):
    temp_max = data['Temperature'].max()
    temp_min = data['Temperature'].min()
    vib_max = data['Vibration'].max()
    vib_min = data['Vibration'].min()
    load_max = data['Load (%)'].max()
    load_min = data['Load (%)'].min()
    num_failures = data['Failure'].sum()

    stats = f"""Estatísticas Gerais:
    Temperatura Máxima: {temp_max:.2f} °C
    Temperatura Mínima: {temp_min:.2f} °C
    Vibração Máxima: {vib_max:.2f}
    Vibração Mínima: {vib_min:.2f}
    Carga Máxima: {load_max:.2f} %
    Carga Mínima: {load_min:.2f} %
    Total de Falhas Detectadas: {num_failures}
    """

    recomendacoes = "Recomendações de Correção:\n"
    if temp_max > limites["temp_max"]:
        recomendacoes += "- Monitorar o sistema de refrigeração para evitar superaquecimento.\n"
    if vib_max > limites["vib_max"]:
        recomendacoes += "- Verificar a estabilidade da máquina e ajustar o balanceamento.\n"
    if load_max > limites["load_max"]:
        recomendacoes += "- Reduzir a carga da máquina para evitar sobrecarga.\n"
    if num_failures > 0:
        recomendacoes += "- Revisar os componentes críticos da máquina.\n"

    return stats, recomendacoes

# Função para exportar relatório em PDF
def exportar_relatorio(stats, recomendacoes):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Título
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, txt="Relatório de Manutenção Preditiva", ln=True, align="C")
        pdf.ln(10)

        # Data e hora
        data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Data e Hora: {data_hora}", ln=True)
        pdf.ln(10)

        # Estatísticas Gerais
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt="Estatísticas Gerais", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=12)
        for linha in stats.split("\n"):
            pdf.cell(200, 10, txt=linha, ln=True)
        pdf.ln(10)

        # Recomendações
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt="Recomendações de Correção", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=12)
        for linha in recomendacoes.split("\n"):
            pdf.cell(200, 10, txt=linha, ln=True)

        # Salvar PDF
        nome_arquivo = f"relatorio_manutencao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(nome_arquivo)
        messagebox.showinfo("Exportação Concluída", f"Relatório salvo como: {nome_arquivo}")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao exportar relatório: {str(e)}")

# Função para atualizar o dashboard
def update_dashboard():
    data = gerar_dados()  # Gerar novos dados com base nos limites
    treinar_modelo(data)  # Treinar o modelo

    # Limpar o gráfico
    fig.clear()

    # Criar subplots
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.plot(data['Time (s)'], data['Temperature'], color='red', label='Temperatura')
    failures_temp_idx = data[data['Failure'] == 1].index
    ax1.scatter(data['Time (s)'][failures_temp_idx], data['Temperature'][failures_temp_idx], color='black', marker='x', s=50, label='Falha (Temperatura)')
    ax1.legend()
    ax1.set_title("Temperatura ao Longo do Tempo")

    ax2.plot(data['Time (s)'], data['Vibration'], color='blue', label='Vibração')
    failures_vib_idx = data[data['Failure'] == 1].index
    ax2.scatter(data['Time (s)'][failures_vib_idx], data['Vibration'][failures_vib_idx], color='orange', marker='x', s=50, label='Falha (Vibração)')
    ax2.legend()
    ax2.set_title("Vibração ao Longo do Tempo")

    ax3.plot(data['Time (s)'], data['Load (%)'], color='green', label='Carga')
    failures_load_idx = data[data['Failure'] == 1].index
    ax3.scatter(data['Time (s)'][failures_load_idx], data['Load (%)'][failures_load_idx], color='purple', marker='x', s=50, label='Falha (Carga)')
    ax3.legend()
    ax3.set_title("Carga ao Longo do Tempo")

    fig.tight_layout()
    canvas.draw()

    # Atualizar feedbacks
    stats, recomendacoes = calcular_estatisticas(data)
    text_estatisticas.delete(1.0, tk.END)
    text_estatisticas.insert(tk.END, stats)
    text_recomendacoes.delete(1.0, tk.END)
    text_recomendacoes.insert(tk.END, recomendacoes)

    # Associar exportação ao relatório
    btn_exportar.config(command=lambda: exportar_relatorio(stats, recomendacoes))

# Funções de hover
def on_hover(event):
    event.widget.config(bg="#0056b3")

def on_leave(event):
    event.widget.config(bg="#007BFF")

def on_hover_export(event):
    event.widget.config(bg="#218838")

def on_leave_export(event):
    event.widget.config(bg="#28a745")

def on_hover_config(event):
    event.widget.config(bg="#5a6268")

def on_leave_config(event):
    event.widget.config(bg="#6c757d")

# Configuração da interface principal
window = tk.Tk()
window.title("Monitoramento de Manutenção Preditiva")
window.geometry("1200x900")

# Gráfico Matplotlib
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Textos para feedbacks
text_estatisticas = tk.Text(window, height=10, wrap='word')
text_estatisticas.pack(fill=tk.X, padx=10, pady=5)
text_recomendacoes = tk.Text(window, height=10, wrap='word')
text_recomendacoes.pack(fill=tk.X, padx=10, pady=5)

# Botão para atualizar o dashboard
btn_update = tk.Button(window, text="Atualizar Dashboard", command=update_dashboard, bg="#007BFF", fg="white", font=("Helvetica", 14), relief="flat", padx=10, pady=5, cursor="hand2")
btn_update.pack(fill=tk.X, padx=400, pady=5)
btn_update.bind("<Enter>", on_hover)
btn_update.bind("<Leave>", on_leave)

# Botão para exportar relatório
btn_exportar = tk.Button(window, text="Exportar Relatório", bg="#28a745", fg="white", font=("Helvetica", 14), relief="flat", padx=10, pady=5, cursor="hand2")
btn_exportar.pack(fill=tk.X, padx=400, pady=5)
btn_exportar.bind("<Enter>", on_hover_export)
btn_exportar.bind("<Leave>", on_leave_export)

# Botão para configurar limites
btn_config = tk.Button(window, text="Configurar Limites", command=abrir_configuracoes, bg="#6c757d", fg="white", font=("Helvetica", 14), relief="flat", padx=10, pady=5, cursor="hand2")
btn_config.pack(fill=tk.X, padx=400, pady=5)
btn_config.bind("<Enter>", on_hover_config)
btn_config.bind("<Leave>", on_leave_config)

# Executar o Tkinter
window.mainloop()
