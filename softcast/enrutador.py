"""Cálculo de la ruta de menor costo con una implementación propia de Dijkstra, O(V²).

Costo de pasar de u a v: peso(u, v) + latencia(v). Se ignoran los nodos Bloqueados
y las aristas inactivas.
"""

import math

from softcast.red import BLOQUEADO, NINGUNA


def calcular_ruta(red, origen, destino):
    """Retorna la lista de índices de la ruta de origen a destino, o NINGUNA.

    Es NINGUNA si el origen o el destino están Bloqueados o si no existe camino.
    """
    if origen == destino:
        raise ValueError("El origen y el destino deben ser nodos distintos.")
    if red.nodos[origen].estado == BLOQUEADO or red.nodos[destino].estado == BLOQUEADO:
        return NINGUNA

    n = len(red.nodos)
    pesos = red.matriz_ponderada()
    distancia = [math.inf] * n
    previo = [None] * n
    visitado = [False] * n
    distancia[origen] = 0.0

    for _ in range(n):
        # Búsqueda lineal del nodo no visitado más cercano; es lo que da O(V²).
        u = None
        for i in range(n):
            if not visitado[i] and distancia[i] < math.inf:
                if u is None or distancia[i] < distancia[u]:
                    u = i
        if u is None or u == destino:
            break
        visitado[u] = True

        for v in range(n):
            # matriz_ponderada() deja en 0 las aristas inactivas.
            if pesos[u, v] == 0 or visitado[v] or red.nodos[v].estado == BLOQUEADO:
                continue
            nueva = distancia[u] + pesos[u, v] + red.nodos[v].latencia
            if nueva < distancia[v]:
                distancia[v] = nueva
                previo[v] = u

    if distancia[destino] == math.inf:
        return NINGUNA

    ruta = [destino]
    while ruta[-1] != origen:
        ruta.append(previo[ruta[-1]])
    ruta.reverse()
    return ruta


def costo_ruta(red, ruta):
    """Suma peso(u, v) + latencia(v) a lo largo de la ruta."""
    pesos = red.matriz_ponderada()
    return sum(
        pesos[u, v] + red.nodos[v].latencia for u, v in zip(ruta, ruta[1:])
    )
