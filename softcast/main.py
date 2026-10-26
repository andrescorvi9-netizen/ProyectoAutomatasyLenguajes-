"""Punto de entrada de SoftCast.

Uso: python -m softcast.main [semilla]
"""

import sys

from softcast.interfaz import main


def leer_semilla(argumentos):
    """Semilla opcional de la línea de comandos; sin ella, se sortea una."""
    if len(argumentos) > 1:
        try:
            return int(argumentos[1])
        except ValueError:
            print(f"La semilla debe ser un número entero: {argumentos[1]}", file=sys.stderr)
            sys.exit(2)
    return None


if __name__ == "__main__":
    main(leer_semilla(sys.argv))
