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
    N0 = 1 / ebfn0_actual
    noise = np.sqrt(N0 / 2) * np.random.normal(0, 1, s.shape)
    r = s + noise
    vr = (r > 0).astype(int)
    
    # Cálculo de Síndrome (Matricial)
    S = np.dot(vr, H_T) % 2
    
    # FILTRADO DE DETECTOR: Máscara booleana para descartar síndromes no nulos
    indices_aceptados = ~np.any(S, axis=1)
    
    # Extraemos solo las palabras que pasaron la validación
    u_aceptadas = u[indices_aceptados]
    vr_aceptadas = vr[indices_aceptados]
    
    palabras_validas = len(u_aceptadas)
    
    if palabras_validas > 0:
        u_recibida_aceptada = vr_aceptadas[:, :k]
        
        errores_bit = np.sum(u_aceptadas != u_recibida_aceptada)
        errores_palabra = np.sum(np.sum(u_aceptadas != u_recibida_aceptada, axis=1) > 0)
        
        # Usamos solo lo que el detector dejó pasar
        P_eb_simulada[i] = errores_bit / (palabras_validas * k)
        P_ep_simulada[i] = errores_palabra / palabras_validas
    else:
        P_eb_simulada[i] = 0
        P_ep_simulada[i] = 0
        
    print(f"  Eb/N0 = {EbN0_dB[i]} dB -> Peb: {P_eb_simulada[i]:.6f} | Pep: {P_ep_simulada[i]:.6f} (Palabras válidas: {palabras_validas}/{M_palabras})")

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