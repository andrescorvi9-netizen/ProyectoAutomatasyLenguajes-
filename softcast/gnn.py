"""Red neuronal convolucional de grafos (GCN) de 2 capas, implementada a mano con PyTorch.

Clasifica cada nodo como Normal, Anómalo o Bloqueado a partir de sus características
y de las de sus vecinos. Arquitectura: GCN 8→16 + ReLU + Dropout 0,2 → GCN 16→16 +
ReLU → Lineal 16→3 + Softmax (467 parámetros).
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from softcast.generador import generar_escenarios
from softcast.red import ESTADOS

CARPETA_PAQUETE = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    # En el ejecutable de PyInstaller los datos se extraen en sys._MEIPASS.
    RUTA_PESOS = Path(sys._MEIPASS) / "softcast" / "modelo" / "pesos_gcn.pt"
else:
    RUTA_PESOS = CARPETA_PAQUETE / "modelo" / "pesos_gcn.pt"
CARPETA_RESULTADOS = CARPETA_PAQUETE.parent / "resultados"

SEMILLA_ENTRENAMIENTO = 42
CANTIDAD_ESCENARIOS = 200
DIVISION = (140, 30, 30)  # entrenamiento, validación, prueba
TASA_APRENDIZAJE = 0.01
DECAIMIENTO_PESOS = 5e-4
EPOCAS_MAXIMAS = 200
PACIENCIA = 20
TAMANO_LOTE = 16


def matriz_normalizada(red):
    """Â normalizada = D̂^(-1/2) (A + I) D̂^(-1/2), con A[i, j] = 1/peso en aristas activas."""
    pesos = red.matriz_ponderada()
    a = np.zeros_like(pesos)
    a[pesos > 0] = 1.0 / pesos[pesos > 0]
    a_gorro = a + np.eye(len(pesos))
    # Un nodo bloqueado solo conserva el lazo propio, así que su grado es 1 y no hay división por 0.
    d_inv_raiz = 1.0 / np.sqrt(a_gorro.sum(axis=1))
    return torch.tensor(d_inv_raiz[:, None] * a_gorro * d_inv_raiz[None, :], dtype=torch.float32)


def entradas(red):
    """Tensores (X de 9×8, Â normalizada de 9×9) de una red."""
    return torch.from_numpy(red.vector_caracteristicas()), matriz_normalizada(red)


def etiquetas(red):
    return torch.tensor([ESTADOS.index(nodo.estado) for nodo in red.nodos])


class CapaGCN(nn.Module):
    """H' = Â · (H · W) + b: cada nodo combina sus características con las de sus vecinos.

    El sesgo se suma después de la agregación, como en la formulación estándar de la GCN.
    """

    def __init__(self, entrada, salida):
        super().__init__()
        self.lineal = nn.Linear(entrada, salida, bias=False)
        self.sesgo = nn.Parameter(torch.zeros(salida))

    def forward(self, h, a):
        return a @ self.lineal(h) + self.sesgo


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.capa1 = CapaGCN(8, 16)
        self.capa2 = CapaGCN(16, 16)
        self.salida = nn.Linear(16, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, a):
        """Retorna logits; la Softmax se aplica en predecir() y, durante el
        entrenamiento, dentro de CrossEntropyLoss."""
        h = self.dropout(torch.relu(self.capa1(x, a)))
        h = torch.relu(self.capa2(h, a))
        return self.salida(h)


def contar_parametros(modelo):
    """Parámetros (pesos + sesgos) por capa."""
    return {
        "GCN 8→16": sum(p.numel() for p in modelo.capa1.parameters()),
        "GCN 16→16": sum(p.numel() for p in modelo.capa2.parameters()),
        "Lineal 16→3": sum(p.numel() for p in modelo.salida.parameters()),
    }


def predecir(modelo, red):
    """Probabilidades (9×3) de cada estado para cada nodo."""
    modelo.eval()
    x, a = entradas(red)
    with torch.no_grad():
        return torch.softmax(modelo(x, a), dim=1)


def clasificar(modelo, red):
    """Actualiza estado_predicho y probabilidades de cada nodo.

    Retorna los índices cuya predicción cambió respecto a la anterior (RF08).
    """
    probabilidades = predecir(modelo, red)
    cambiados = []
    for nodo, fila in zip(red.nodos, probabilidades):
        nuevo = ESTADOS[int(fila.argmax())]
        if nodo.estado_predicho is not None and nuevo != nodo.estado_predicho:
            cambiados.append(nodo.indice)
        nodo.estado_predicho = nuevo
        nodo.probabilidades = [round(float(p), 4) for p in fila]
    return cambiados


def dividir_escenarios(escenarios):
    entrenamiento, validacion, _ = DIVISION
    return (
        escenarios[:entrenamiento],
        escenarios[entrenamiento : entrenamiento + validacion],
        escenarios[entrenamiento + validacion :],
    )


def _apilar(escenarios):
    x, a = zip(*(entradas(red) for red in escenarios))
    y = [etiquetas(red) for red in escenarios]
    return torch.stack(x), torch.stack(a), torch.stack(y)


def pesos_por_clase(y):
    """Peso inverso a la frecuencia: total / (clases × casos de la clase)."""
    conteo = torch.bincount(y.flatten(), minlength=len(ESTADOS)).float()
    return y.numel() / (len(ESTADOS) * conteo)


def entrenar(escenarios_entrenamiento, escenarios_validacion, semilla=SEMILLA_ENTRENAMIENTO):
    """Entrena con Adam y parada temprana sobre la pérdida de validación.

    Retorna el modelo con los pesos de la mejor época y un registro del entrenamiento.
    """
    torch.manual_seed(semilla)
    modelo = GCN()
    x_ent, a_ent, y_ent = _apilar(escenarios_entrenamiento)
    x_val, a_val, y_val = _apilar(escenarios_validacion)

    pesos_clase = pesos_por_clase(y_ent)
    criterio = nn.CrossEntropyLoss(weight=pesos_clase)
    optimizador = torch.optim.Adam(
        modelo.parameters(), lr=TASA_APRENDIZAJE, weight_decay=DECAIMIENTO_PESOS
    )
    barajador = torch.Generator().manual_seed(semilla)

    historial = []
    mejor_perdida = float("inf")
    mejor_estado = None
    mejor_epoca = 0
    epocas_sin_mejora = 0
    motivo = f"se alcanzó el máximo de {EPOCAS_MAXIMAS} épocas"

    for epoca in range(1, EPOCAS_MAXIMAS + 1):
        modelo.train()
        orden = torch.randperm(len(x_ent), generator=barajador)
        suma_perdida = 0.0
        for inicio in range(0, len(orden), TAMANO_LOTE):
            lote = orden[inicio : inicio + TAMANO_LOTE]
            optimizador.zero_grad()
            logits = modelo(x_ent[lote], a_ent[lote])
            perdida = criterio(logits.reshape(-1, 3), y_ent[lote].reshape(-1))
            perdida.backward()
            optimizador.step()
            suma_perdida += perdida.item() * len(lote)

        modelo.eval()
        with torch.no_grad():
            logits_val = modelo(x_val, a_val).reshape(-1, 3)
            perdida_val = criterio(logits_val, y_val.reshape(-1)).item()
            exactitud_val = (logits_val.argmax(dim=1) == y_val.reshape(-1)).float().mean().item()
        historial.append(
            {
                "epoca": epoca,
                "perdida_entrenamiento": suma_perdida / len(x_ent),
                "perdida_validacion": perdida_val,
                "exactitud_validacion": exactitud_val,
            }
        )

        if perdida_val < mejor_perdida:
            mejor_perdida = perdida_val
            mejor_estado = copy.deepcopy(modelo.state_dict())
            mejor_epoca = epoca
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= PACIENCIA:
                motivo = (
                    f"parada temprana: {PACIENCIA} épocas sin mejorar la pérdida de "
                    f"validación desde la época {mejor_epoca}"
                )
                break

    modelo.load_state_dict(mejor_estado)
    registro = {
        "pesos_clase": dict(zip(ESTADOS, (round(float(p), 4) for p in pesos_clase))),
        "epoca_final": len(historial),
        "mejor_epoca": mejor_epoca,
        "motivo_detencion": motivo,
        "historial": historial,
    }
    return modelo, registro


def calcular_metricas(reales, predichas):
    """Exactitud, matriz de confusión (filas: real, columnas: predicho) y métricas por clase."""
    n = len(ESTADOS)
    confusion = np.zeros((n, n), dtype=int)
    for real, predicha in zip(reales, predichas):
        confusion[real, predicha] += 1

    por_clase = {}
    for c, estado in enumerate(ESTADOS):
        verdaderos = confusion[c, c]
        predichos = confusion[:, c].sum()
        soporte = confusion[c, :].sum()
        precision = verdaderos / predichos if predichos else 0.0
        recall = verdaderos / soporte if soporte else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        por_clase[estado] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "soporte": int(soporte),
        }

    return {
        "exactitud": round(float(np.trace(confusion) / confusion.sum()), 4),
        "matriz_confusion": confusion.tolist(),
        "por_clase": por_clase,
    }


def metricas(modelo, escenarios):
    reales, predichas = [], []
    for red in escenarios:
        reales.extend(etiquetas(red).tolist())
        predichas.extend(predecir(modelo, red).argmax(dim=1).tolist())
    return calcular_metricas(reales, predichas)


def guardar_curva(historial, ruta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epocas = [fila["epoca"] for fila in historial]
    figura, eje = plt.subplots(figsize=(7, 4))
    eje.plot(epocas, [f["perdida_entrenamiento"] for f in historial], label="Entrenamiento")
    eje.plot(epocas, [f["perdida_validacion"] for f in historial], label="Validación")
    eje.set_xlabel("Época")
    eje.set_ylabel("Pérdida (entropía cruzada ponderada)")
    eje.set_title("Curva de pérdida de la GCN")
    eje.legend()
    eje.grid(alpha=0.3)
    figura.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def cargar_o_entrenar(ruta_pesos=RUTA_PESOS):
    """Carga los pesos si el archivo existe; si no, entrena y los guarda.

    Retorna (modelo, registro); registro es None cuando los pesos se cargaron.
    """
    modelo = GCN()
    if ruta_pesos.exists():
        modelo.load_state_dict(torch.load(ruta_pesos, weights_only=True))
        modelo.eval()
        return modelo, None

    entrenamiento, validacion, _ = dividir_escenarios(generar_escenarios(CANTIDAD_ESCENARIOS))
    modelo, registro = entrenar(entrenamiento, validacion)
    ruta_pesos.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), ruta_pesos)
    modelo.eval()
    return modelo, registro


if __name__ == "__main__":
    modelo, registro = cargar_o_entrenar()
    parametros = contar_parametros(modelo)

    print("Parámetros por capa")
    for capa, cantidad in parametros.items():
        print(f"  {capa:<12} {cantidad:>4}")
    print(f"  {'Total':<12} {sum(parametros.values()):>4} (esperado 467)\n")

    if registro is None:
        print(f"Pesos cargados desde {RUTA_PESOS}; no se reentrenó.\n")
    else:
        print("Pesos por clase (inversos a la frecuencia en entrenamiento)")
        for estado, peso in registro["pesos_clase"].items():
            print(f"  {estado:<10} {peso:.4f}")
        print(f"\n{'Época':>5} {'Pérd. entr.':>12} {'Pérd. val.':>11} {'Exact. val.':>12}")
        for fila in registro["historial"]:
            print(
                f"{fila['epoca']:>5} {fila['perdida_entrenamiento']:>12.4f} "
                f"{fila['perdida_validacion']:>11.4f} {fila['exactitud_validacion']:>12.2%}"
            )
        print(f"\nSe detuvo en la época {registro['epoca_final']}: {registro['motivo_detencion']}.")
        print(f"Pesos guardados de la época {registro['mejor_epoca']} en {RUTA_PESOS}\n")
        guardar_curva(registro["historial"], CARPETA_RESULTADOS / "curva_perdida.png")

    _, _, prueba = dividir_escenarios(generar_escenarios(CANTIDAD_ESCENARIOS))
    resultado = metricas(modelo, prueba)

    print(f"Evaluación en {len(prueba)} escenarios de prueba ({len(prueba) * 9} nodos)")
    print(f"  Exactitud: {resultado['exactitud']:.2%} (meta ≥ 85 %)")
    print("  Matriz de confusión (filas: real; columnas: predicho)")
    print(f"  {'':<10}" + "".join(f"{e:>11}" for e in ESTADOS))
    for estado, fila in zip(ESTADOS, resultado["matriz_confusion"]):
        print(f"  {estado:<10}" + "".join(f"{v:>11}" for v in fila))
    print(f"\n  {'Clase':<10} {'Precisión':>10} {'Recall':>8} {'F1':>8} {'Soporte':>8}")
    for estado, m in resultado["por_clase"].items():
        print(
            f"  {estado:<10} {m['precision']:>10.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {m['soporte']:>8}"
        )

    if registro is not None:
        CARPETA_RESULTADOS.mkdir(exist_ok=True)
        with open(CARPETA_RESULTADOS / "metricas.json", "w", encoding="utf-8") as archivo:
            json.dump(
                {"parametros": parametros, "prueba": resultado, "entrenamiento": registro},
                archivo,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nMétricas guardadas en {CARPETA_RESULTADOS / 'metricas.json'}")
