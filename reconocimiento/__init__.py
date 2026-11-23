"""Sistema de reconocimiento facial EBGM (detector YuNet + EBGM + BD .csv).

    from reconocimiento import SistemaReconocimiento
    sis = SistemaReconocimiento()
    sis.identificar("foto.jpg")

Los imports son diferidos para que `python -m reconocimiento.sistema` (o
`.detector`) no cargue el módulo dos veces.
"""

_EXPORTA = {
    "SistemaReconocimiento": "sistema", "ErrorSistema": "sistema", "ErrorCalidad": "sistema",
    "DetectorCaras": "detector", "ErrorDeteccion": "detector",
}
__all__ = list(_EXPORTA)


def __getattr__(nombre):
    if nombre in _EXPORTA:
        from importlib import import_module
        return getattr(import_module(f".{_EXPORTA[nombre]}", __name__), nombre)
    raise AttributeError(f"module {__name__!r} has no attribute {nombre!r}")
