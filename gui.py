from fpdf import FPDF
import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import filedialog
import os
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import logging

# Configuração Base
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
plt.style.use('dark_background')

class PredictiveGuardApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PredictiveGuard - Monitoramento Preditivo Industrial")
        self.geometry("1400x850")
        
        # Limites operacionais padrão alinhados com a engenharia industrial
        self.limites = {"temp_max": 85.0, "vib_max": 45.0, "load_max": 90.0}
        
        # Variáveis de Streaming e Controle
        self.is_streaming = False
        self.tempo_atual = 0
        self.janela_visualizacao = 60 
        self.dados_live = pd.DataFrame(columns=['Time', 'Temperature', 'Vibration', 'Load', 'Failure'])
        
        # Controle de Estabilidade de Tela (Smoothing e Latching)
        self.historico_risco = []
        self.timer_alertas = {"risco": 0, "temp": 0, "vib": 0, "load": 0}
        
        # Inicializa Banco de Dados, recupera histórico prévio e treina a IA
        self.inicializar_banco()
        self.modelo, self.top_feature, self.top_importance = self.inicializar_modelo_ia()
        
        self.construir_interface()
        self.carregar_historico_inicial()

        # Garante o fechamento seguro dos processos ao fechar a janela
        self.protocol("WM_DELETE_WINDOW", self.fechar_aplicacao)

    def inicializar_banco(self):
        self.conn = sqlite3.connect("historico_preditivo.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS leituras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                temperatura REAL,
                vibracao REAL,
                carga REAL,
                risco_falha REAL,
                falha_detectada INTEGER
            )
        ''')
        self.conn.commit()
        logging.info("Banco de dados SQLite conectado com sucesso.")

    def carregar_historico_inicial(self):
        """Carrega os últimos 60 registros do banco para que a tela não inicie vazia."""
        try:
            query = "SELECT timestamp, temperatura, vibracao, carga, falha_detectada FROM leituras ORDER BY id DESC LIMIT 60"
            df_hist = pd.read_sql_query(query, self.conn)
            
            if not df_hist.empty:
                df_hist = df_hist.sort_values(by="id").reset_index(drop=True)
                # Renomeia para o padrão da interface
                df_hist.rename(columns={'temperatura': 'Temperature', 'vibracao': 'Vibration', 'carga': 'Load', 'falha_detectada': 'Failure'}, inplace=True)
                
                # Reatribui o tempo sequencial para visualização
                df_hist['Time'] = range(len(df_hist))
                self.dados_live = df_hist[['Time', 'Temperature', 'Vibration', 'Load', 'Failure']]
                self.tempo_atual = len(self.dados_live)
                
                self.atualizar_graficos_live()
                logging.info(f"Histórico carregado do banco: {len(self.dados_live)} registros restaurados na tela.")
            else:
                self.atualizar_graficos_vazios()
        except Exception as e:
            logging.error(f"Erro ao carregar histórico inicial: {e}")
            self.atualizar_graficos_vazios()

    def inicializar_modelo_ia(self):
        # Simulação calibrada para comportamento industrial realista
        temp = np.random.normal(65, 4, 2000)
        vib = np.random.normal(25, 6, 2000)
        load = np.random.uniform(40, 85, 2000)
        amb_temp = np.random.uniform(22, 35, 2000)
        hum = np.random.uniform(40, 70, 2000)
        age = np.random.uniform(1, 5, 2000)

        falhas = np.where((temp > self.limites["temp_max"]) | (vib > self.limites["vib_max"]) | (load > self.limites["load_max"]), 1, 0)
        
        X_train = pd.DataFrame({'Temp': temp, 'Vib': vib, 'Load': load, 'AmbTemp': amb_temp, 'Hum': hum, 'Age': age})
        y_train = falhas
        
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_
        features = ['Temperatura', 'Vibração', 'Carga', 'Temp. Ambiente', 'Umidade', 'Idade']
        idx_max = np.argmax(importances)
        
        return model, features[idx_max], importances[idx_max] * 100

    def construir_interface(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)

        self.lbl_logo = ctk.CTkLabel(self.sidebar_frame, text="PredictiveGuard", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_logo.pack(padx=20, pady=(30, 10))

        self.lbl_status = ctk.CTkLabel(self.sidebar_frame, text="STATUS: AGUARDANDO", font=ctk.CTkFont(size=16, weight="bold"), fg_color="#6c757d", text_color="white", corner_radius=8, height=35)
        self.lbl_status.pack(padx=20, pady=(0, 20), fill="x")

        self.btn_toggle = ctk.CTkButton(self.sidebar_frame, text="Iniciar Monitoramento", command=self.toggle_streaming, font=("Helvetica", 14, "bold"), height=45)
        self.btn_toggle.pack(padx=20, pady=10, fill="x")

        self.btn_config = ctk.CTkButton(self.sidebar_frame, text="Configurar Limites", command=self.abrir_configuracoes, fg_color="#4d4d4d", hover_color="#333333", font=("Helvetica", 14, "bold"), height=45)
        self.btn_config.pack(padx=20, pady=10, fill="x")

        self.btn_exportar = ctk.CTkButton(self.sidebar_frame, text="Exportar Relatório", command=self.exportar_relatorio, fg_color="#28a745", hover_color="#218838", font=("Helvetica", 14, "bold"), height=45)
        self.btn_exportar.pack(padx=20, pady=10, fill="x")

        ctk.CTkLabel(self.sidebar_frame, text="Leitura Atual (Real-Time)", font=ctk.CTkFont(size=16, weight="bold")).pack(padx=20, pady=(30, 5), anchor="w")
        self.text_estatisticas = ctk.CTkTextbox(self.sidebar_frame, height=180, font=("Consolas", 13), fg_color="#1a1a1a", text_color="#dce4ee", corner_radius=8, border_width=1, border_color="#333333")
        self.text_estatisticas.pack(padx=20, pady=(0, 10), fill="x")
        self.text_estatisticas.configure(state="disabled")

        ctk.CTkLabel(self.sidebar_frame, text="Ações Recomendadas", font=ctk.CTkFont(size=16, weight="bold")).pack(padx=20, pady=(10, 5), anchor="w")
        self.text_recomendacoes = ctk.CTkTextbox(self.sidebar_frame, height=180, font=("Consolas", 13), fg_color="#1a1a1a", text_color="#ffcc00", corner_radius=8, wrap="word", border_width=1, border_color="#333333")
        self.text_recomendacoes.pack(padx=20, pady=(0, 20), fill="both", expand=True)
        self.text_recomendacoes.configure(state="disabled")

        self.main_frame = ctk.CTkFrame(self, fg_color="#242424", corner_radius=0)
        self.main_frame.pack(side="right", fill="both", expand=True)

        self.fig = plt.figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('#242424')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=20, pady=20)
        
        self.stats_atuais = "Nenhuma leitura ativa."
        self.recomendacoes_atuais = "Sistema pronto para iniciar."

    def abrir_configuracoes(self):
        config_window = ctk.CTkToplevel(self)
        config_window.title("Configurações de Limites")
        config_window.geometry("350x400")
        config_window.attributes('-topmost', 'true')
        config_window.resizable(False, False)

        frame_centro = ctk.CTkFrame(config_window, fg_color="transparent")
        frame_centro.pack(expand=True)

        ctk.CTkLabel(frame_centro, text="Temperatura Máxima (°C):", font=("Helvetica", 14)).pack(pady=(10, 5))
        entry_temp = ctk.CTkEntry(frame_centro, font=("Helvetica", 14), width=200, justify="center")
        entry_temp.insert(0, str(self.limites["temp_max"]))
        entry_temp.pack(pady=5)

        ctk.CTkLabel(frame_centro, text="Vibração Máxima:", font=("Helvetica", 14)).pack(pady=(15, 5))
        entry_vib = ctk.CTkEntry(frame_centro, font=("Helvetica", 14), width=200, justify="center")
        entry_vib.insert(0, str(self.limites["vib_max"]))
        entry_vib.pack(pady=5)

        ctk.CTkLabel(frame_centro, text="Carga Máxima (%):", font=("Helvetica", 14)).pack(pady=(15, 5))
        entry_load = ctk.CTkEntry(frame_centro, font=("Helvetica", 14), width=200, justify="center")
        entry_load.insert(0, str(self.limites["load_max"]))
        entry_load.pack(pady=5)

        def salvar_configuracoes():
            try:
                self.limites["temp_max"] = float(entry_temp.get())
                self.limites["vib_max"] = float(entry_vib.get())
                self.limites["load_max"] = float(entry_load.get())
                messagebox.showinfo("Configurações Salvas", f"Limites atualizados com sucesso!\n\nTemp Max: {self.limites['temp_max']}°C\nVib Max: {self.limites['vib_max']}\nCarga Max: {self.limites['load_max']}%")
                config_window.destroy()
            except ValueError:
                messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")

        btn_salvar = ctk.CTkButton(frame_centro, text="Salvar Limites", command=salvar_configuracoes, width=200, height=40, font=("Helvetica", 14, "bold"))
        btn_salvar.pack(pady=(30, 10))

    def toggle_streaming(self):
        self.is_streaming = not self.is_streaming
        if self.is_streaming:
            self.btn_toggle.configure(text="Pausar Monitoramento", fg_color="#dc3545", hover_color="#c82333")
            self.lbl_status.configure(text="STATUS: MONITORANDO", fg_color="#17a2b8")
            self.loop_streaming()
        else:
            self.btn_toggle.configure(text="Iniciar Monitoramento", fg_color="#1f538d", hover_color="#14375e")
            self.lbl_status.configure(text="STATUS: PAUSADO", fg_color="#6c757d")

    def loop_streaming(self):
        if not self.is_streaming:
            return

        # 1. Geração de dados industriais calibrados com oscilações reais
        temp = np.random.normal(68, 3.5)
        vib = np.random.normal(26, 5)
        load = np.random.uniform(45, 88)
        amb_temp = np.random.uniform(22, 32)
        hum = np.random.uniform(40, 65)
        age = np.random.uniform(2, 6)

        # 2. Avaliação de Falha com base estrita nos LIMITES CONFIGURADOS PELO USUÁRIO
        falha = 1 if (temp > self.limites["temp_max"]) or (vib > self.limites["vib_max"]) or (load > self.limites["load_max"]) else 0

        features = pd.DataFrame({'Temp': [temp], 'Vib': [vib], 'Load': [load], 'AmbTemp': [amb_temp], 'Hum': [hum], 'Age': [age]})
        prob_instantanea = self.modelo.predict_proba(features)[0][1] * 100 

        # Suavização da probabilidade (Média Móvel dos últimos 5 segundos)
        self.historico_risco.append(prob_instantanea)
        if len(self.historico_risco) > 5:
            self.historico_risco.pop(0)
        probabilidade_suavizada = sum(self.historico_risco) / len(self.historico_risco)

        # 3. Salvar no banco de dados SQLite
        agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute('''
            INSERT INTO leituras (timestamp, temperatura, vibracao, carga, risco_falha, falha_detectada)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (agora, temp, vib, load, probabilidade_suavizada, int(falha)))
        self.conn.commit()

        # 4. Adiciona ao DataFrame live da interface
        novo_dado = pd.DataFrame({'Time': [self.tempo_atual], 'Temperature': [temp], 'Vibration': [vib], 'Load': [load], 'Failure': [falha]})
        self.dados_live = pd.concat([self.dados_live, novo_dado], ignore_index=True)

        if len(self.dados_live) > self.janela_visualizacao:
            self.dados_live = self.dados_live.iloc[1:].reset_index(drop=True)

        self.tempo_atual += 1

        # 5. Atualiza a Tela
        self.atualizar_graficos_live()
        self.atualizar_textos_live(temp, vib, load, falha, probabilidade_suavizada)

        # Continua o loop de segundo em segundo de forma segura
        if self.is_streaming:
            self.after(1000, self.loop_streaming)

    def atualizar_graficos_vazios(self):
        self.fig.clear()
        for i in range(1, 4):
            ax = self.fig.add_subplot(3, 1, i)
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#a3a3a3')
            ax.grid(color='#333333', linestyle='-', linewidth=0.5)
            for spine in ax.spines.values(): spine.set_color('#333333')
        
        # Legendas Dinâmicas baseadas nos limites atuais
        self.fig.add_subplot(3, 1, 1).set_title(f"Temperatura Operacional (°C) - Limite: {self.limites['temp_max']}°C", color='white', pad=5)
        self.fig.add_subplot(3, 1, 2).set_title(f"Níveis de Vibração - Limite: {self.limites['vib_max']}", color='white', pad=5)
        self.fig.add_subplot(3, 1, 3).set_title(f"Carga da Máquina (%) - Limite: {self.limites['load_max']}%", color='white', pad=5)
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def atualizar_graficos_live(self):
        self.fig.clear()
        ax1 = self.fig.add_subplot(3, 1, 1)
        ax2 = self.fig.add_subplot(3, 1, 2)
        ax3 = self.fig.add_subplot(3, 1, 3)

        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#a3a3a3')
            ax.grid(color='#333333', linestyle='-', linewidth=0.5)
            for spine in ax.spines.values(): spine.set_color('#333333')
            if not self.dados_live.empty:
                ax.set_xlim(max(0, self.dados_live['Time'].max() - self.janela_visualizacao), max(self.janela_visualizacao, self.dados_live['Time'].max()))

        if not self.dados_live.empty:
            # Plot Temperatura
            ax1.plot(self.dados_live['Time'], self.dados_live['Temperature'], color='#ff4a4a', linewidth=1.5)
            falhas_temp = self.dados_live[self.dados_live['Failure'] == 1]
            if not falhas_temp.empty:
                ax1.scatter(falhas_temp['Time'], falhas_temp['Temperature'], color='white', marker='x')
            
            # Linha de Limite Crítico no Gráfico
            ax1.axhline(y=self.limites['temp_max'], color='#ff4a4a', linestyle='--', linewidth=1, alpha=0.7, label='Limite Configurado')

            # Plot Vibração
            ax2.plot(self.dados_live['Time'], self.dados_live['Vibration'], color='#4a90e2', linewidth=1.5)
            if not falhas_temp.empty:
                ax2.scatter(falhas_temp['Time'], falhas_temp['Vibration'], color='orange', marker='x')
            ax2.axhline(y=self.limites['vib_max'], color='#4a90e2', linestyle='--', linewidth=1, alpha=0.7)

            # Plot Carga
            ax3.plot(self.dados_live['Time'], self.dados_live['Load'], color='#50e3c2', linewidth=1.5)
            if not falhas_temp.empty:
                ax3.scatter(falhas_temp['Time'], falhas_temp['Load'], color='white', marker='x')
            ax3.axhline(y=self.limites['load_max'], color='#50e3c2', linestyle='--', linewidth=1, alpha=0.7)

        # Legendas refeltindo perfeitamente os limites atuais configurados
        ax1.set_title(f"Temperatura Operacional (°C) | Teto Max: {self.limites['temp_max']}°C", color='white', pad=5)
        ax2.set_title(f"Níveis de Vibração | Teto Max: {self.limites['vib_max']}", color='white', pad=5)
        ax3.set_title(f"Carga da Máquina (%) | Teto Max: {self.limites['load_max']}%", color='white', pad=5)

        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def atualizar_textos_live(self, temp, vib, load, falha, prob_falha):
        falhas_janela = int(self.dados_live['Failure'].sum()) if not self.dados_live.empty else 0

        stats = f"Tempo de Op.: {self.tempo_atual}s\n"
        stats += f"Temp. Atual: {temp:.2f} °C\n"
        stats += f"Vibração Atual: {vib:.2f}\n"
        stats += f"Carga Atual: {load:.2f} %\n"
        stats += "-"*20 + "\n"
        stats += f"Risco de Falha (Média): {prob_falha:.1f}%\n"
        stats += f"Anomalias na Janela: {falhas_janela}\n"
        stats += f"Fator Crítico IA: {self.top_feature} ({self.top_importance:.1f}%)"

        # Retenção de Alarme (Latching)
        if prob_falha > 70 or falha == 1: self.timer_alertas["risco"] = 5
        if temp > self.limites["temp_max"]: self.timer_alertas["temp"] = 5
        if vib > self.limites["vib_max"]: self.timer_alertas["vib"] = 5
        if load > self.limites["load_max"]: self.timer_alertas["load"] = 5

        recomendacoes = ""
        if self.timer_alertas["risco"] > 0:
            recomendacoes += "RISCO CRÍTICO (IA):\nPreparar interrupção de emergência.\n\n"
            self.timer_alertas["risco"] -= 1
            
        if self.timer_alertas["temp"] > 0:
            recomendacoes += f"ALERTA TEMP (>{self.limites['temp_max']}°C):\nChecar trocador de calor e refrigeração.\n\n"
            self.timer_alertas["temp"] -= 1
            
        if self.timer_alertas["vib"] > 0:
            recomendacoes += f"ALERTA VIB (>{self.limites['vib_max']}):\nVerificar balanceamento e mancais.\n\n"
            self.timer_alertas["vib"] -= 1
            
        if self.timer_alertas["load"] > 0:
            recomendacoes += f"ALERTA CARGA (>{self.limites['load_max']}%):\nReduzir taxa de alimentação.\n\n"
            self.timer_alertas["load"] -= 1
        
        if recomendacoes == "":
            recomendacoes = "OPERAÇÃO ESTÁVEL\nParâmetros dentro dos limites normais."

        if falhas_janela == 0 and prob_falha < 50:
            self.lbl_status.configure(text="STATUS: NORMAL", fg_color="#28a745", text_color="white")
        elif falhas_janela < 5 and prob_falha < 75:
            self.lbl_status.configure(text="STATUS: ATENÇÃO", fg_color="#ffc107", text_color="black")
        else:
            self.lbl_status.configure(text="STATUS: CRÍTICO", fg_color="#dc3545", text_color="white")

        self.text_estatisticas.configure(state="normal")
        self.text_estatisticas.delete(1.0, ctk.END)
        self.text_estatisticas.insert(ctk.END, stats)
        self.text_estatisticas.configure(state="disabled")

        self.text_recomendacoes.configure(state="normal")
        self.text_recomendacoes.delete(1.0, ctk.END)
        self.text_recomendacoes.insert(ctk.END, recomendacoes)
        self.text_recomendacoes.configure(state="disabled")

        self.stats_atuais = stats
        self.recomendacoes_atuais = recomendacoes

    def exportar_relatorio(self):
        if self.dados_live.empty:
            messagebox.showwarning("Aviso", "Não há dados suficientes para exportar.")
            return

        caminho_sugerido = f"relatorio_manutencao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Salvar Relatório Como",
            initialfile=caminho_sugerido
        )

        if not filepath:
            return

        try:
            img_path = "temp_dash.png"
            self.fig.savefig(img_path, facecolor='#242424', bbox_inches='tight')

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", style="B", size=16)
            pdf.cell(200, 10, txt="Relatório de Manutenção Preditiva (Live)", ln=True, align="C")
            pdf.ln(5)
            
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Data e Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True)
            pdf.ln(5)
            
            pdf.image(img_path, x=10, w=190)
            pdf.ln(10)

            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(200, 10, txt="Leitura Atual no Momento da Exportação", ln=True)
            pdf.set_font("Arial", size=12)
            for linha in self.stats_atuais.split("\n"):
                linha_pdf = linha.encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(200, 7, txt=linha_pdf, ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(200, 10, txt="Alertas e Recomendações Vigentes", ln=True)
            pdf.set_font("Arial", size=12)
            for linha in self.recomendacoes_atuais.split("\n"):
                linha_pdf = linha.replace("⚠️ ", "").replace("🚨 ", "").replace("✅ ", "")
                linha_pdf = linha_pdf.encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(200, 7, txt=linha_pdf, ln=True)

            pdf.output(filepath)
            if os.path.exists(img_path):
                os.remove(img_path)

            logging.info("Relatório exportado: %s", filepath)
            caminho_absoluto = os.path.abspath(filepath)
            messagebox.showinfo("Exportação Concluída", f"Relatório salvo com sucesso em:\n\n{caminho_absoluto}")
            
        except Exception as e:
            logging.error("Erro ao exportar relatório: %s", str(e))
            messagebox.showerror("Erro", f"Erro ao exportar relatório: {str(e)}")

    def fechar_aplicacao(self):
        """Para a thread de streaming e fecha a conexão com o banco de forma segura."""
        self.is_streaming = False
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
                logging.info("Conexão com banco de dados encerrada com segurança.")
            except Exception:
                pass
        self.destroy()

if __name__ == "__main__":
    app = PredictiveGuardApp()
    app.mainloop()