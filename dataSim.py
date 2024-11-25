import numpy as np
import pandas as pd

# Função para simular os dados
def gerar_dados():
    # Variáveis básicas
    time = np.arange(0, 1000, 1)  # Tempo em segundos
    temperature = np.random.normal(70, 5, 1000)  # Temperatura operacional
    vibration = np.random.normal(30, 10, 1000)  # Vibração operacional

    # Novos fatores
    load = np.random.uniform(50, 100, 1000)  # Carga da máquina (%)
    ambient_temp = np.random.uniform(20, 40, 1000)  # Temperatura ambiente (°C)
    humidity = np.random.uniform(30, 80, 1000)  # Umidade ambiente (%)
    machine_age = np.random.uniform(1, 10, 1000)  # Idade da máquina (anos)

    # Regra para simular falhas
    failures = np.where(
        (temperature > 80) |
        (vibration > 50) |
        ((load > 90) & (temperature > 75)) |
        ((ambient_temp > 35) & (humidity > 70)) |
        (machine_age > 8),
        1,  # Falha ocorre
        0   # Sem falha
    )

    # Criar DataFrame com os dados simulados
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

# Exemplo de uso
if __name__ == "__main__":
    dados = gerar_dados()
    print(dados.head())  # Visualizar os primeiros dados gerados
