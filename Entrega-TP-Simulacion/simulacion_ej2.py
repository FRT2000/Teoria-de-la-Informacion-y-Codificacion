import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, comb

# =================================================================
# 1. PARÁMETROS DEL SISTEMA
# =================================================================
n = 14  # Largo de palabra de código
k = 10  # Largo de palabra de fuente

M_bits = 1000000
M_palabras = M_bits // k

# Vector de relación Señal a Ruido (Eb/N0) en Decibeles (de 0 a 8 dB)
EbN0_dB = np.arange(0, 9, 1)

# EbfN0 = Cociente energía de bit de fuente sobre densidad espectral de ruido 
# (Convertimos los dB a escala lineal)
EbfN0 = 10**(EbN0_dB / 10)

# =================================================================
# 2. DEFINICIÓN DE MATRICES G Y H^T (Código 14,10)
# =================================================================
I_k = np.eye(k, dtype=int)
P = np.array([
    [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0],
    [1, 1, 0, 1], [1, 1, 1, 0]
])
G = np.hstack((I_k, P))
I_nk = np.eye(n - k, dtype=int)
H_T = np.vstack((P, I_nk))

print("Paso 1 Completado: Entorno configurado.")

# =================================================================
# 3. GENERACIÓN DE FUENTE, CODIFICACIÓN Y MODULACIÓN
# =================================================================
print("Generando bits de fuente y codificando...")

# u = Bits de fuente originales
u = np.random.randint(0, 2, (M_palabras, k))

# v = Palabra de código (vector fila) codificada sistemáticamente
v = np.dot(u, G) % 2

# Es = Energía de símbolo de canal
# Ebf = Energía de bit de fuente (Asumimos Ebf = 1 para normalizar)
A = np.sqrt(k / n)

# s = Palabra de código modulada en amplitud
s = (2 * v - 1) * A

print("Paso 2 Completado: Bits generados y modulados.")

# =================================================================
# 4. SIMULACIÓN DEL CANAL AWGN Y RECEPTOR (Decisión Dura)
# =================================================================
print("Iniciando simulación del canal AWGN (de 0 a 8 dB)...")

P_eb_simulada = np.zeros(len(EbN0_dB))
P_ep_simulada = np.zeros(len(EbN0_dB))

for i, ebfn0_actual in enumerate(EbfN0):
    # N0 = Densidad espectral de ruido deseado
    N0 = 1 / ebfn0_actual
    
    # noise = Ruido AWGN
    noise = np.sqrt(N0 / 2) * np.random.normal(0, 1, s.shape)
    
    # r = Palabra de código recibida
    r = s + noise
    
    # vr = Palabra de código demodulada (detección dura)
    vr = (r > 0).astype(int)
    
    # Cálculo de Síndrome (Matricial) - En un detector puro sirve para pedir retransmisión
    # S = np.dot(vr, H_T) % 2
    
    # Decodificador: Detección (SIN CORRECCIÓN)
    # Como no corregimos, simplemente tomamos los primeros k bits de la palabra recibida
    u_recibida = vr[:, :k]
    
    # Extracción de los bits de información y conteo de errores crudos del canal
    errores_bit = np.sum(u != u_recibida)
    errores_palabra = np.sum(np.sum(u != u_recibida, axis=1) > 0)
    
    P_eb_simulada[i] = errores_bit / M_bits
    P_ep_simulada[i] = errores_palabra / M_palabras
    
    print(f"  Eb/N0 = {EbN0_dB[i]} dB -> Peb: {P_eb_simulada[i]:.5f} | Pep: {P_ep_simulada[i]:.5f}")

# =================================================================
# 5. CÁLCULO TEÓRICO Y GRÁFICOS
# =================================================================

print("Generando gráficos...")

P_eb_teorica_BPSK = 0.5 * erfc(np.sqrt(EbfN0))

# Curva Teórica del Código (14,10) como Detector (td = 2)
td = 2
p_canal = 0.5 * erfc(np.sqrt((k/n) * EbfN0))

P_ep_teorica_det = np.zeros(len(EbfN0))
# Sumatoria desde i = td+1 hasta n
for i in range(td + 1, n + 1):
    P_ep_teorica_det += comb(n, i) * (p_canal**i) * ((1 - p_canal)**(n - i))

plt.figure(figsize=(10, 6))

plt.semilogy(EbN0_dB, P_eb_teorica_BPSK, 'k--', linewidth=2, label='Teórica sin codificar (BPSK)')
# Teórica del detector
plt.semilogy(EbN0_dB, P_ep_teorica_det, 'm-', linewidth=2, label='Pep Teórica (Detector)')

# Simuladas
plt.semilogy(EbN0_dB, P_eb_simulada, 'bo-', linewidth=2, label='Peb Simulada (Detector)')
plt.semilogy(EbN0_dB, P_ep_simulada, 'rx-', linewidth=2, label='Pep Simulada (Detector)')

plt.title('Desempeño del Código de Bloque Lineal (14,10) - Detector')
plt.xlabel('Relación Señal a Ruido por bit de fuente, $E_{bf}/N_0$ (dB)')
plt.ylabel('Probabilidad de Error ($P_e$)')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend()
plt.ylim(1e-5, 1)
plt.xlim(0, 8)
plt.show()

print("¡Simulación Completa!")