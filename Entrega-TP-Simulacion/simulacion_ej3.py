import numpy as np
from PIL import Image
import heapq
from collections import Counter

def construir_arbol_huffman(probabilidades):
    """Construye el árbol de Huffman y retorna el diccionario de códigos."""
    # Cola de prioridad para armar el árbol: [peso, [simbolo, codigo]]
    heap = [[p, [simb, ""]] for simb, p in probabilidades.items()]
    heapq.heapify(heap)

    # Si por algún motivo hay un solo símbolo
    if len(heap) == 1:
        return {heap[0][1][0]: "0"}

    # Reducción iterativa de Huffman
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        # Asignamos '0' a la rama izquierda y '1' a la derecha
        for par in lo[1:]:
            par[1] = '0' + par[1]
        for par in hi[1:]:
            par[1] = '1' + par[1]
            
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
    # Extraemos el diccionario de la raíz del árbol
    arbol = heapq.heappop(heap)[1:]
    # Ordenamos por longitud del código
    arbol.sort(key=lambda x: (len(x[1]), x[0]))
    return {simb: cod for simb, cod in arbol}

def analizar_fuente_extendida(pixeles, orden):
    """Agrupa, calcula probabilidades, aplica Huffman y muestra métricas."""
    print(f"\n{'-'*50}")
    print(f"ANALIZANDO FUENTE EXTENDIDA DE ORDEN {orden}")
    print(f"{'-'*50}")
    
    # Recortar píxeles sobrantes para que armen bloques exactos
    longitud_util = len(pixeles) - (len(pixeles) % orden)
    datos = pixeles[:longitud_util]
    
    # Agrupar en bloques (ej: "01", "11" para orden 2)
    bloques = ["".join(map(str, datos[i:i+orden])) for i in range(0, longitud_util, orden)]
    total_bloques = len(bloques)
    
    # 1. Estimación de probabilidades
    conteo = Counter(bloques)
    probabilidades = {simb: cuenta / total_bloques for simb, cuenta in conteo.items()}
    
    # 2. Algoritmo de Huffman
    diccionario = construir_arbol_huffman(probabilidades)
    
    # 3. Cálculo de métricas
    # Largo de la secuencia original (cantidad de bits de fuente)
    largo_secuencia_original = len(datos)
    
    # Largo total de la secuencia codificada (frecuencia absoluta * largo del código)
    largo_secuencia_codificada = sum(conteo[simb] * len(cod) for simb, cod in diccionario.items())
    
    # Largo promedio = (Largo total codificado) / (Cantidad de bits de fuente originales)
    largo_promedio = largo_secuencia_codificada / largo_secuencia_original
    
    # Tasa de compresión = (Secuencia original) / (Secuencia codificada)
    tasa_compresion = largo_secuencia_original / largo_secuencia_codificada 
    
    print(f"Largo Promedio:\t\t{largo_promedio:.4f} bits de código / bit de fuente")
    print(f"Tasa de Compresión:\t{tasa_compresion:.4f}")
    
    print("\nDiccionario Huffman obtenido:")
    for simb, cod in diccionario.items():
        print(f"  Bloque '{simb}': \tCódigo '{cod}' \t(Probabilidad: {probabilidades[simb]:.4f})")

# =================================================================
# BLOQUE PRINCIPAL
# =================================================================
if __name__ == "__main__":
    try:
        # Abrir imagen y forzar formato blanco y negro (1 bit)
        img = Image.open('logo FI.tif').convert('1')
        
        # Convertir a matriz numpy y aplastar a 1D
        matriz = np.array(img)
        # Aseguramos que queden estrictamente unos (blancos) y ceros (negros)
        pixeles = (matriz > 0).astype(int).flatten()
        
        print(f"\nImagen cargada. Total de píxeles: {len(pixeles)}")
        
        # Comprobar las probabilidades de la fuente de Orden 1
        prob_blanco = np.sum(pixeles == 1) / len(pixeles)
        prob_negro = np.sum(pixeles == 0) / len(pixeles)
        print(f"Probabilidad de Blanco (1):\t{prob_blanco:.4f}")
        print(f"Probabilidad de Negro (0):\t{prob_negro:.4f}")
        
        # Ejecutar análisis para orden 2 y 3
        analizar_fuente_extendida(pixeles, orden=2)
        analizar_fuente_extendida(pixeles, orden=3)
        
    except FileNotFoundError:
        print("ERROR: No se encontró 'logo FI.tif' dentro de la carpeta.")
    except ImportError:
        print("ERROR: Faltan librerías. Ejecuta: pip install Pillow numpy")