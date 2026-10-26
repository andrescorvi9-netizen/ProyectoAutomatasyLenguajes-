import pytest

from softcast.automata import (
    MENSAJE_CADENA_VACIA,
    MENSAJE_SIMBOLOS_SISTEMA,
    NOMBRES_ESTADOS,
    evaluar,
    mensaje_cadena_incompleta,
    preparar_cadena,
    validar_cadena,
)
from softcast.generador import generar_red
from softcast.red import ANOMALO, NINGUNA, NORMAL

CASOS_OBLIGATORIOS = [
    ("sadkf", True, ["q0", "q1", "q2", "q3", "q2", "q6"], None),
    ("saf", True, ["q0", "q1", "q2", "q6"], None),
    ("sadkdkf", True, ["q0", "q1", "q2", "q3", "q2", "q3", "q2", "q6"], None),
    ("sadrrkf", True, ["q0", "q1", "q2", "q3", "q4", "q5", "q2", "q6"], None),
    ("sadrrrkf", False, ["q0", "q1", "q2", "q3", "q4", "q5", "q7"], (6, "r")),
    ("saddkf", False, ["q0", "q1", "q2", "q3", "q7"], (4, "d")),
    ("sbadkf", False, ["q0", "q1", "q7"], (2, "b")),
    ("dkf", False, ["q0", "q7"], (1, "d")),
    ("sadkfd", False, ["q0", "q1", "q2", "q3", "q2", "q6", "q7"], (6, "d")),
]


@pytest.mark.parametrize("cadena, aceptada, recorrido, rechazo", CASOS_OBLIGATORIOS)
def test_casos_obligatorios(cadena, aceptada, recorrido, rechazo):
    assert evaluar(cadena) == (aceptada, recorrido, rechazo)


def test_cadena_incompleta_se_rechaza_al_final():
    assert evaluar("sad") == (False, ["q0", "q1", "q2", "q3"], (4, None))


def test_nombres_de_estados_segun_tabla_15():
    assert NOMBRES_ESTADOS == {
        "q0": "Inactivo",
        "q1": "Solicitando",
        "q2": "Conectado",
        "q3": "Transmitiendo",
        "q4": "En espera (1.er retardo)",
        "q5": "En espera (2.º retardo)",
        "q6": "Finalizado",
        "q7": "Fallido",
    }


@pytest.mark.parametrize(
    "cadena, mensaje",
    [
        ("", "Rechazada: la cadena terminó en q0 (Inactivo); falta la solicitud s."),
        ("s", "Rechazada: la cadena terminó en q1 (Solicitando); falta la aceptación a."),
        ("sa", "Rechazada: la cadena terminó en q2 (Conectado); falta finalizar con f."),
        ("sad", "Rechazada: la cadena terminó en q3 (Transmitiendo); "
                "falta confirmar con k y finalizar con f."),
        ("sadr", "Rechazada: la cadena terminó en q4 (En espera); "
                 "falta confirmar con k y finalizar con f."),
        ("sadrr", "Rechazada: la cadena terminó en q5 (En espera); "
                  "falta confirmar con k y finalizar con f."),
    ],
)
def test_mensaje_de_cadena_incompleta(cadena, mensaje):
    aceptada, recorrido, rechazo = evaluar(cadena)
    assert not aceptada and rechazo == (len(cadena) + 1, None)
    assert mensaje_cadena_incompleta(recorrido[-1]) == mensaje


@pytest.mark.parametrize("cadena", ["sadrkf", "sbadkf", "sadkfb"])
def test_validacion_rechaza_r_y_b(cadena):
    assert validar_cadena(cadena) == (cadena, MENSAJE_SIMBOLOS_SISTEMA)


def test_validacion_rechaza_simbolo_fuera_de_sigma():
    assert validar_cadena("saxkf") == ("saxkf", "Símbolo no válido: x")


def test_validacion_acepta_simbolos_de_usuario():
    assert validar_cadena("sadkdkf") == ("sadkdkf", None)


def test_validacion_quita_espacios_de_los_extremos():
    assert validar_cadena("  sadkf \t") == ("sadkf", None)


def test_validacion_espacio_interno_es_invalido():
    assert validar_cadena(" sad kf ") == ("sad kf", "Símbolo no válido: espacio")


@pytest.mark.parametrize("cadena", ["", "   "])
def test_validacion_cadena_vacia(cadena):
    assert validar_cadena(cadena) == ("", MENSAJE_CADENA_VACIA)


def _red_con_estados(anomalos):
    red = generar_red(7)
    for indice in range(len(red.nodos)):
        red.cambiar_estado(indice, ANOMALO if indice in anomalos else NORMAL)
    return red


def test_preparar_sin_ruta_inserta_b():
    cadena = preparar_cadena("sadkf", NINGUNA, _red_con_estados(set()))
    assert cadena == "sbadkf"
    assert evaluar(cadena)[0] is False


def test_preparar_cuenta_anomalos_incluidos_extremos():
    # Ruta IoT1 → PC2 → PC4 → IoT3 con origen y destino Anómalos.
    red = _red_con_estados({4, 6})
    cadena = preparar_cadena("sadkdkf", [4, 1, 3, 6], red)
    assert cadena == "sadrrkdrrkf"
    assert evaluar(cadena)[0] is True


def test_preparar_con_tres_anomalos_rechaza():
    red = _red_con_estados({4, 1, 6})
    cadena = preparar_cadena("sadkf", [4, 1, 3, 6], red)
    assert cadena == "sadrrrkf"
    assert evaluar(cadena)[0] is False


def test_preparar_ruta_normal_no_modifica():
    assert preparar_cadena("sadkf", [4, 1, 3, 6], _red_con_estados(set())) == "sadkf"
