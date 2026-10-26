import networkx as nx
import pytest

from softcast.enrutador import calcular_ruta, costo_ruta
from softcast.generador import generar_escenarios
from softcast.red import (
    BLOQUEADO,
    IDS,
    NINGUNA,
    NORMAL,
    PARES_ARISTAS,
    TIPO_POR_INDICE,
    Arista,
    Nodo,
    Red,
)

PC1, PC2, PC3, PC4, IOT1, IOT2, IOT3, SRV_L, SRV_G = range(9)


def red_controlada(bloqueados=(), peso=5.0, latencia=10.0):
    """Red con todos los pesos y latencias iguales, para que la ruta esperada sea única."""
    nodos = [
        Nodo(i, IDS[i], TIPO_POR_INDICE[i], 20.0, 30.0, latencia, NORMAL)
        for i in range(9)
    ]
    red = Red(nodos, [Arista(o, d, peso) for o, d in PARES_ARISTAS])
    for indice in bloqueados:
        red.cambiar_estado(indice, BLOQUEADO)
    return red


def test_iot1_a_iot3_con_srv_l_bloqueado():
    red = red_controlada(bloqueados=[SRV_L])
    ruta = calcular_ruta(red, IOT1, IOT3)
    assert ruta == [IOT1, PC2, PC4, IOT3]
    assert costo_ruta(red, ruta) == 3 * (5.0 + 10.0)


def test_iot1_a_iot3_sin_bloqueos_pasa_por_srv_l():
    assert calcular_ruta(red_controlada(), IOT1, IOT3) == [IOT1, SRV_L, IOT3]


def test_latencia_alta_desvia_la_ruta():
    red = red_controlada()
    red.nodos[SRV_L].latencia = 500.0
    assert calcular_ruta(red, IOT1, IOT3) == [IOT1, PC2, PC4, IOT3]


@pytest.mark.parametrize("bloqueado, origen, destino", [(IOT1, IOT1, IOT3), (IOT3, IOT1, IOT3)])
def test_extremo_bloqueado_retorna_ninguna(bloqueado, origen, destino):
    assert calcular_ruta(red_controlada(bloqueados=[bloqueado]), origen, destino) is NINGUNA


def test_sin_camino_retorna_ninguna():
    # IoT1 solo se conecta con PC2 y SRV-L.
    red = red_controlada(bloqueados=[PC2, SRV_L])
    assert calcular_ruta(red, IOT1, IOT3) is NINGUNA


def test_origen_igual_a_destino_es_error():
    with pytest.raises(ValueError):
        calcular_ruta(red_controlada(), PC1, PC1)


def _costo_networkx(red, origen, destino):
    grafo = nx.DiGraph()
    for arista in red.aristas:
        if arista.activa:
            for u, v in ((arista.origen, arista.destino), (arista.destino, arista.origen)):
                grafo.add_edge(u, v, costo=arista.peso + red.nodos[v].latencia)
    try:
        return nx.dijkstra_path_length(grafo, origen, destino, weight="costo")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def test_coincide_con_networkx_en_200_escenarios():
    for red in generar_escenarios(200):
        for origen in range(9):
            for destino in range(9):
                if origen == destino:
                    continue
                ruta = calcular_ruta(red, origen, destino)
                esperado = _costo_networkx(red, origen, destino)
                if esperado is None or BLOQUEADO in (red.nodos[origen].estado, red.nodos[destino].estado):
                    assert ruta is NINGUNA
                else:
                    assert costo_ruta(red, ruta) == pytest.approx(esperado)
                    assert all(red.nodos[i].estado != BLOQUEADO for i in ruta)
