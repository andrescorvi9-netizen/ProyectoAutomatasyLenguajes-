"""Generación aleatoria (distribución uniforme) de redes y escenarios de entrenamiento."""

import random

from softcast.red import (
    ANOMALO,
    BLOQUEADO,
    ESTADOS,
    IDS,
    NORMAL,
    PARES_ARISTAS,
    TIPO_POR_INDICE,
    Arista,
    Nodo,
    Red,
)

PESOS_ESTADO_INICIAL = (70, 20, 10)  # Normal, Anómalo, Bloqueado (%)

RANGOS_TRAFICO = {
    "PC": {NORMAL: (10, 50), ANOMALO: (51, 100), BLOQUEADO: (0, 0)},
    "IoT": {NORMAL: (5, 30), ANOMALO: (31, 60), BLOQUEADO: (0, 0)},
    "Servidor": {NORMAL: (40, 80), ANOMALO: (81, 150), BLOQUEADO: (0, 0)},
}
RANGOS_CPU = {NORMAL: (10, 60), ANOMALO: (61, 100), BLOQUEADO: (0, 5)}
RANGOS_LATENCIA = {NORMAL: (5, 50), ANOMALO: (150, 500), BLOQUEADO: (1000, 1000)}
RANGO_PESO = (1, 10)

SEMILLA_ESCENARIOS = 42


def _grupo_trafico(tipo):
    # SRV-L y SRV-G comparten el rango de tráfico de servidores.
    return "Servidor" if tipo.startswith("SRV") else tipo


def _uniforme(aleatorio, rango):
    # Con un rango degenerado (0, 0) o (1000, 1000) el resultado es exacto.
    return round(aleatorio.uniform(*rango), 1)


def regenerar_caracteristicas(nodo, estado, aleatorio=random):
    """Asigna tráfico, CPU y latencia uniformes dentro del rango de `estado` (RF06).

    Solo modifica las características; el estado y las aristas los actualiza
    Red.cambiar_estado(), que debe llamarse antes.
    """
    nodo.trafico = _uniforme(aleatorio, RANGOS_TRAFICO[_grupo_trafico(nodo.tipo)][estado])
    nodo.cpu = _uniforme(aleatorio, RANGOS_CPU[estado])
    nodo.latencia = _uniforme(aleatorio, RANGOS_LATENCIA[estado])


def generar_red(semilla=None):
    """Crea una red con estados, características y pesos aleatorios.

    La misma semilla produce siempre la misma red (RNF04). Sin semilla se sortea una
    y queda guardada en red.semilla para poder reproducirla.
    """
    if semilla is None:
        # Desde 1000 para no repetir las semillas de entrenamiento (42 a 241).
        semilla = random.randrange(1000, 1_000_000)
    aleatorio = random.Random(semilla)

    nodos = []
    for indice, (id_nodo, tipo) in enumerate(zip(IDS, TIPO_POR_INDICE)):
        estado = aleatorio.choices(ESTADOS, weights=PESOS_ESTADO_INICIAL)[0]
        nodo = Nodo(indice, id_nodo, tipo, 0, 0, 0, estado)
        regenerar_caracteristicas(nodo, estado, aleatorio)
        nodos.append(nodo)

    aristas = [
        Arista(origen, destino, _uniforme(aleatorio, RANGO_PESO))
        for origen, destino in PARES_ARISTAS
    ]
    return Red(nodos, aristas, semilla)


def generar_escenarios(cantidad=200, semilla=SEMILLA_ESCENARIOS):
    """Genera `cantidad` redes independientes; el escenario i usa la semilla `semilla + i`."""
    return [generar_red(semilla + i) for i in range(cantidad)]


if __name__ == "__main__":
    red = generar_red(7)

    print(f"Red generada con semilla {red.semilla}\n")
    print(f"{'Id':<6} {'Tipo':<6} {'Tráfico':>8} {'CPU':>6} {'Latencia':>9}  Estado")
    for nodo in red.nodos:
        print(
            f"{nodo.id:<6} {nodo.tipo:<6} {nodo.trafico:>8.1f} {nodo.cpu:>6.1f} "
            f"{nodo.latencia:>9.1f}  {nodo.estado}"
        )

    activas = sum(arista.activa for arista in red.aristas)
    print(f"\nAristas activas: {activas}; inactivas: {len(red.aristas) - activas}")
    print(f"Suma de grados de matriz_base: {red.matriz_base.sum()} (esperado 42)")
    print(f"Grado de SRV-L: {red.matriz_base[7].sum()} (esperado 8)")
    filas, columnas = red.vector_caracteristicas().shape
    print(f"Forma de vector_caracteristicas(): {filas} × {columnas} (esperado 9 × 8)")
