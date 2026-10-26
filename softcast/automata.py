"""Autómata finito determinista que valida la secuencia de una transmisión.

M = (Q, Σ, δ, q0, F), con Q = {q0..q7}, Σ = {s, a, d, k, r, f, b}, F = {q6}.
q7 es el estado sumidero. Lenguaje reconocido: sa(d(ε + r + rr)k)*f
"""

from softcast.red import ANOMALO, NINGUNA

ESTADOS = ("q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7")
NOMBRES_ESTADOS = {
    "q0": "Inactivo",
    "q1": "Solicitando",
    "q2": "Conectado",
    "q3": "Transmitiendo",
    "q4": "En espera (1.er retardo)",
    "q5": "En espera (2.º retardo)",
    "q6": "Finalizado",
    "q7": "Fallido",
}
# Lo que le falta a una cadena que se agotó en cada estado no final.
FALTANTES = {
    "q0": "falta la solicitud s",
    "q1": "falta la aceptación a",
    "q2": "falta finalizar con f",
    "q3": "falta confirmar con k y finalizar con f",
    "q4": "falta confirmar con k y finalizar con f",
    "q5": "falta confirmar con k y finalizar con f",
}
ALFABETO = ("s", "a", "d", "k", "r", "f", "b")
ESTADO_INICIAL = "q0"
ESTADOS_FINALES = {"q6"}
SUMIDERO = "q7"

# Solo se listan las transiciones que no van al sumidero.
TRANSICIONES = {
    ("q0", "s"): "q1",
    ("q1", "a"): "q2",
    ("q2", "d"): "q3",
    ("q2", "f"): "q6",
    ("q3", "k"): "q2",
    ("q3", "r"): "q4",
    ("q4", "k"): "q2",
    ("q4", "r"): "q5",
    ("q5", "k"): "q2",
}

SIMBOLOS_USUARIO = ("s", "a", "d", "k", "f")
MENSAJE_SIMBOLOS_SISTEMA = (
    "Los símbolos r y b los agrega el sistema según el estado de la ruta; "
    "no deben escribirse."
)
MENSAJE_CADENA_VACIA = "Ingrese una cadena de eventos."


def delta(estado, simbolo):
    """Función de transición δ: Q × Σ → Q."""
    if simbolo not in ALFABETO:
        raise ValueError(f"Símbolo no válido: {simbolo}")
    return TRANSICIONES.get((estado, simbolo), SUMIDERO)


def evaluar(cadena):
    """Retorna (aceptada, recorrido, rechazo).

    El recorrido empieza en q0 y termina en el primer q7, como en la Tabla 18 del
    Diseño. rechazo es (posición 1-based, símbolo) o None si la cadena se aceptó.
    Si la cadena se agota en un estado no final, rechazo es (len(cadena) + 1, None).
    """
    estado = ESTADO_INICIAL
    recorrido = [estado]
    for posicion, simbolo in enumerate(cadena, start=1):
        estado = delta(estado, simbolo)
        recorrido.append(estado)
        if estado == SUMIDERO:
            return False, recorrido, (posicion, simbolo)
    if estado in ESTADOS_FINALES:
        return True, recorrido, None
    return False, recorrido, (len(cadena) + 1, None)


def mensaje_cadena_incompleta(estado):
    """Mensaje para una cadena que se agotó en `estado` sin llegar a q6."""
    # En q4 y q5 el mensaje solo dice "En espera", sin el número de retardo.
    nombre = NOMBRES_ESTADOS[estado].split(" (")[0]
    return f"Rechazada: la cadena terminó en {estado} ({nombre}); {FALTANTES[estado]}."


def validar_cadena(cadena):
    """Revisa la cadena escrita por el usuario antes de preparar_cadena().

    Retorna (cadena sin espacios en los extremos, mensaje). El mensaje es None si la
    cadena es válida, o el texto que debe mostrarse en la interfaz.
    """
    cadena = cadena.strip()
    if not cadena:
        return cadena, MENSAJE_CADENA_VACIA
    if "r" in cadena or "b" in cadena:
        return cadena, MENSAJE_SIMBOLOS_SISTEMA
    for simbolo in cadena:
        if simbolo not in SIMBOLOS_USUARIO:
            nombre = "espacio" if simbolo == " " else simbolo
            return cadena, f"Símbolo no válido: {nombre}"
    return cadena, None


def preparar_cadena(cadena, ruta, red):
    """Inserta los símbolos que dependen de la ruta calculada por el enrutador.

    Sin ruta (NINGUNA) se inserta 'b' después de la primera 's'. Con ruta, después
    de cada 'd' se insertan tantas 'r' como nodos Anómalos tenga la ruta, contando
    origen y destino.
    """
    if ruta is NINGUNA:
        return cadena.replace("s", "sb", 1)
    anomalos = sum(1 for indice in ruta if red.nodos[indice].estado == ANOMALO)
    return cadena.replace("d", "d" + "r" * anomalos)
