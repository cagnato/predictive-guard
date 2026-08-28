from fpdf import FPDF
import customtkinter as ctk
import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import logging

# Configuração do Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração do visual moderno (Tema Escuro/Azul)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Estilo para deixar os gráficos do Matplotlib mais bonitos
plt.style.use('ggplot')

# Variáveis globais para os limites
limites = {
    "temp_max": 80,
    "vib_max": 50,
    "load_max": 90
}

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

    config_window = ctk.CTkToplevel(window)
    config_window.title("Configurações de Limites")
    config_window.geometry("400x350")
    config_window.attributes('-topmost', 'true')

    ctk.CTkLabel(config_window, text="Temperatura Máxima (°C):", font=("Helvetica", 14)).pack(pady=(20, 5))
    entry_temp = ctk.CTkEntry(config_window, font=("Helvetica", 14))
    entry_temp.insert(0, str(limites["temp_max"]))
    entry_temp.pack(pady=5)

    ctk.CTkLabel(config_window, text="Vibração Máxima:", font=("Helvetica", 14)).pack(pady=5)
    entry_vib = ctk.CTkEntry(config_window, font=("Helvetica", 14))
    entry_vib.insert(0, str(limites["vib_max"]))
    entry_vib.pack(pady=5)

    ctk.CTkLabel(config_window, text="Carga Máxima (%):", font=("Helvetica", 14)).pack(pady=5)
    entry_load = ctk.CTkEntry(config_window, font=("Helvetica", 14))
    entry_load.insert(0, str(limites["load_max"]))
    entry_load.pack(pady=5)

    btn_salvar = ctk.CTkButton(config_window, text="Salvar Limites", command=salvar_configuracoes)
    btn_salvar.pack(pady=20)

def gerar_dados():
    time = np.arange(0, 1000, 1)
    temperature = np.random.normal(70, 5, 1000)
    vibration = np.random.normal(30, 10, 1000)
    load = np.random.uniform(50, 100, 1000)
    ambient_temp = np.random.uniform(20, 40, 1000)
    humidity = np.random.uniform(30, 80, 1000)
    machine_age = np.random.uniform(1, 10, 1000)

    failures = np.where(
        (temperature > limites["temp_max"]) |
        (vibration > limites["vib_max"]) |
        (load > limites["load_max"]),
        1, 0
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
    
    # Registro do Log
    logging.info("Dados simulados com sucesso. Total de registros: %d", len(data))
    
    return data

def treinar_modelo(data):
    X = data[['Temperature', 'Vibration', 'Load (%)', 'Ambient Temp (°C)', 'Humidity (%)', 'Machine Age (years)']]
    y = data['Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info("Modelo treinado com precisão: %.2f", accuracy)
    return model

def calcular_estatisticas(data):
    temp_max, temp_min = data['Temperature'].max(), data['Temperature'].min()
    vib_max, vib_min = data['Vibration'].max(), data['Vibration'].min()
    load_max, load_min = data['Load (%)'].max(), data['Load (%)'].min()
    num_failures = data['Failure'].sum()

    stats = f"""ESTATÍSTICAS GERAIS:
Temperatura Máxima: {temp_max:.2f} °C
Temperatura Mínima: {temp_min:.2f} °C
Vibração Máxima: {vib_max:.2f}
Vibração Mínima: {vib_min:.2f}
Carga Máxima: {load_max:.2f} %
Carga Mínima: {load_min:.2f} %
Total de Falhas Detectadas: {num_failures}"""

    recomendacoes = "RECOMENDAÇÕES DE CORREÇÃO:\n"
    if temp_max > limites["temp_max"]:
        recomendacoes += "- Monitorar sistema de refrigeração.\n"
    if vib_max > limites["vib_max"]:
        recomendacoes += "- Ajustar balanceamento da máquina.\n"
    if load_max > limites["load_max"]:
        recomendacoes += "- Reduzir a carga para evitar sobrecarga.\n"
    if num_failures > 0:
        recomendacoes += "- Revisar os componentes críticos.\n"

    return stats, recomendacoes

def exportar_relatorio(stats, recomendacoes):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, txt="Relatório de Manutenção Preditiva", ln=True, align="C")
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Data e Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True)
        pdf.ln(10)

        for linha in stats.split("\n"):
            pdf.cell(200, 10, txt=linha, ln=True)
        pdf.ln(10)

        for linha in recomendacoes.split("\n"):
            pdf.cell(200, 10, txt=linha, ln=True)

        nome_arquivo = f"relatorio_manutencao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(nome_arquivo)
        logging.info("Relatório exportado: %s", nome_arquivo)
        messagebox.showinfo("Exportação Concluída", f"Relatório salvo como: {nome_arquivo}")
    except Exception as e:
        logging.error("Erro ao exportar relatório: %s", str(e))
        messagebox.showerror("Erro", f"Erro ao exportar relatório: {str(e)}")

def update_dashboard():
    data = gerar_dados()
    treinar_modelo(data)

    fig.clear()
    fig.patch.set_facecolor('#2b2b2b') # Fundo escuro para os gráficos combinarem com o UI

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors='white')
        ax.title.set_color('white')

    ax1.plot(data['Time (s)'], data['Temperature'], color='#ff4a4a', label='Temperatura')
    failures_temp_idx = data[data['Failure'] == 1].index
    ax1.scatter(data['Time (s)'][failures_temp_idx], data['Temperature'][failures_temp_idx], color='white', marker='x', label='Falha')
    ax1.legend()
    ax1.set_title("Temperatura")

    ax2.plot(data['Time (s)'], data['Vibration'], color='#4a90e2', label='Vibração')
    failures_vib_idx = data[data['Failure'] == 1].index
    ax2.scatter(data['Time (s)'][failures_vib_idx], data['Vibration'][failures_vib_idx], color='orange', marker='x', label='Falha')
    ax2.legend()
    ax2.set_title("Vibração")

    ax3.plot(data['Time (s)'], data['Load (%)'], color='#50e3c2', label='Carga')
    failures_load_idx = data[data['Failure'] == 1].index
    ax3.scatter(data['Time (s)'][failures_load_idx], data['Load (%)'][failures_load_idx], color='white', marker='x', label='Falha')
    ax3.legend()
    ax3.set_title("Carga")

    fig.tight_layout()
    canvas.draw()

    stats, recomendacoes = calcular_estatisticas(data)
    text_estatisticas.delete(1.0, ctk.END)
    text_estatisticas.insert(ctk.END, stats)
    text_recomendacoes.delete(1.0, ctk.END)
    text_recomendacoes.insert(ctk.END, recomendacoes)

    btn_exportar.configure(command=lambda: exportar_relatorio(stats, recomendacoes))

# Configuração da interface principal com CustomTkinter
window = ctk.CTk()
window.title("PredictiveGuard - Monitoramento Preditivo")
window.geometry("1200x900")

# Gráfico Matplotlib
fig = plt.figure(figsize=(10, 6))
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=20, pady=20)

# Container para os textos
frame_textos = ctk.CTkFrame(window, fg_color="transparent")
frame_textos.pack(fill=ctk.X, padx=20, pady=10)

text_estatisticas = ctk.CTkTextbox(frame_textos, height=120, font=("Consolas", 12))
text_estatisticas.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=(0, 10))

text_recomendacoes = ctk.CTkTextbox(frame_textos, height=120, font=("Consolas", 12))
text_recomendacoes.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=(10, 0))

# Container para os botões
frame_botoes = ctk.CTkFrame(window, fg_color="transparent")
frame_botoes.pack(fill=ctk.X, padx=20, pady=20)

btn_update = ctk.CTkButton(frame_botoes, text="Atualizar Dashboard", command=update_dashboard, font=("Helvetica", 14, "bold"), height=40)
btn_update.pack(side=ctk.LEFT, expand=True, padx=10)

btn_exportar = ctk.CTkButton(frame_botoes, text="Exportar Relatório", fg_color="#28a745", hover_color="#218838", font=("Helvetica", 14, "bold"), height=40)
btn_exportar.pack(side=ctk.LEFT, expand=True, padx=10)

btn_config = ctk.CTkButton(frame_botoes, text="Configurar Limites", command=abrir_configuracoes, fg_color="#6c757d", hover_color="#5a6268", font=("Helvetica", 14, "bold"), height=40)
btn_config.pack(side=ctk.LEFT, expand=True, padx=10)

window.mainloop()