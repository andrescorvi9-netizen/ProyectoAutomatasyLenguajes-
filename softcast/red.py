"""Modelo de la red SoftCast: grafo no dirigido ponderado de 9 nodos y 21 aristas."""

from dataclasses import dataclass, field

import numpy as np

NORMAL = "Normal"
ANOMALO = "Anómalo"
BLOQUEADO = "Bloqueado"
ESTADOS = (NORMAL, ANOMALO, BLOQUEADO)

TIPOS = ("PC", "IoT", "SRV-L", "SRV-G")

# Valor que devuelve el enrutador cuando no existe ruta entre origen y destino.
NINGUNA = None

IDS = ("PC1", "PC2", "PC3", "PC4", "IoT1", "IoT2", "IoT3", "SRV-L", "SRV-G")
TIPO_POR_INDICE = ("PC", "PC", "PC", "PC", "IoT", "IoT", "IoT", "SRV-L", "SRV-G")

# Pares (origen, destino) por índice, agrupados como en el documento de Diseño.
PARES_ARISTAS = (
    # PC–IoT
    (1, 4), (2, 5), (3, 6),
    # PC–PC
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    # PC–SRV-L e IoT–SRV-L
    (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    # PC–SRV-G y SRV-L–SRV-G
    (0, 8), (1, 8), (2, 8), (3, 8), (7, 8),
)

# Cotas fijas de la normalización min-max; el grado máximo es el de SRV-L.
TRAFICO_MAXIMO = 150.0
CPU_MAXIMA = 100.0
LATENCIA_MAXIMA = 1000.0
GRADO_MAXIMO = 8.0


@dataclass
class Nodo:
    indice: int
    id: str
    tipo: str
    trafico: float
    cpu: float
    latencia: float
    estado: str
    estado_predicho: str | None = None
    probabilidades: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class Arista:
    origen: int
    destino: int
    peso: float
    activa: bool = True


class Red:
    """Grafo de la red con el estado de cada nodo y la actividad de cada arista."""

    def __init__(self, nodos, aristas, semilla=None):
        self.nodos = nodos
        self.aristas = aristas
        self.semilla = semilla
        n = len(nodos)
        self.matriz_base = np.zeros((n, n), dtype=int)
        for arista in aristas:
            self.matriz_base[arista.origen, arista.destino] = 1
            self.matriz_base[arista.destino, arista.origen] = 1
        self._actualizar_aristas()

    def _actualizar_aristas(self):
        # Una arista solo está activa si ninguno de sus extremos está Bloqueado.
        for arista in self.aristas:
            arista.activa = (
                self.nodos[arista.origen].estado != BLOQUEADO
                and self.nodos[arista.destino].estado != BLOQUEADO
            )

    def cambiar_estado(self, indice, estado):
        """Asigna el nuevo estado al nodo y recalcula qué aristas quedan activas.

        No regenera las características; eso lo hace generador.regenerar_caracteristicas().
        """
        if estado not in ESTADOS:
            raise ValueError(f"Estado no válido: {estado}")
        self.nodos[indice].estado = estado
        self._actualizar_aristas()

    def matriz_activa(self):
        """Matriz de adyacencia 0/1 con fila y columna en 0 para los nodos bloqueados."""
        n = len(self.nodos)
        matriz = np.zeros((n, n), dtype=int)
        for arista in self.aristas:
            if arista.activa:
                matriz[arista.origen, arista.destino] = 1
                matriz[arista.destino, arista.origen] = 1
        return matriz

    def matriz_ponderada(self):
        """Matriz con el peso (ms) de cada arista activa y 0 donde no hay conexión."""
        n = len(self.nodos)
        matriz = np.zeros((n, n), dtype=float)
        for arista in self.aristas:
            if arista.activa:
                matriz[arista.origen, arista.destino] = arista.peso
                matriz[arista.destino, arista.origen] = arista.peso
        return matriz

    def vector_caracteristicas(self):
        """Matriz 9×8 de entrada a la GCN.

        Columnas: tráfico, CPU y latencia normalizados; one-hot del tipo (PC, IoT,
        SRV-L, SRV-G); grado activo / 8.
        """
        grados = self.matriz_activa().sum(axis=1)
        filas = []
        for nodo in self.nodos:
            one_hot = [1.0 if nodo.tipo == tipo else 0.0 for tipo in TIPOS]
            filas.append(
                [
                    nodo.trafico / TRAFICO_MAXIMO,
                    nodo.cpu / CPU_MAXIMA,
                    nodo.latencia / LATENCIA_MAXIMA,
                    *one_hot,
                    grados[nodo.indice] / GRADO_MAXIMO,
                ]
            )
        return np.array(filas, dtype=np.float32)
