import numpy as np
import matplotlib.pyplot as plt

# 1. Generar vector de tiempo y una señal de prueba
t = np.linspace(0, 1, 100)
senal = np.sin(2 * np.pi * 5 * t)

# 2. Generar ruido gaussiano (similar al AWGN)
ruido = np.random.normal(0, 0.5, 100)
senal_ruidosa = senal + ruido

# 3. Graficar los resultados
plt.figure(figsize=(10, 5))
plt.plot(t, senal, label="Señal original (sin ruido)", color="blue", linewidth=2)
plt.plot(t, senal_ruidosa, label="Señal recibida (con ruido)", color="red", alpha=0.6)

plt.title("Prueba de Entorno: Canal Ruidoso")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

# Mostrar el gráfico por pantalla
plt.show()

print("¡Si estás viendo el gráfico y este mensaje, tu entorno está 100% operativo!")