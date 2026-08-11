"""Estilo visual compartido por los notebooks de EDA y el dashboard.

Tener esto en un módulo, y no repetido en cada notebook, es lo que hace que las
figuras del README, las de los notebooks y las del dashboard se vean como un
mismo sistema y no como cuatro personas distintas.

La paleta categórica está validada para daltonismo: la separación mínima entre
slots adyacentes es ΔE 9.1 en OKLab (el umbral es 8) bajo simulación de
protanopia. El orden de los slots ES el mecanismo de seguridad — se asignan
siempre en orden, nunca en ciclo.

Dos slots (aqua y amarillo) quedan por debajo de 3:1 de contraste contra el
fondo claro, así que las series que los usen llevan siempre leyenda o etiqueta
visible: el color nunca es el único canal que transmite identidad.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Paleta categórica (asignar EN ORDEN, nunca ciclar) -------------------

SERIES = [
    "#2a78d6",  # 1 azul
    "#eb6834",  # 2 naranja
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 amarillo
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 verde
    "#4a3aa7",  # 7 violeta
    "#e34948",  # 8 rojo
]

# --- Tinta y superficies --------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- Estados (nunca se reutilizan como color de serie) --------------------

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# --- Colores semánticos de los regímenes del dataset ----------------------
#
# Los tres regímenes aparecen en muchas figuras. Fijar su color aquí evita que
# "data drift" sea naranja en un gráfico y verde en otro.
REGIME_COLORS = {
    "baseline": "#2a78d6",
    "data_drift": "#eda100",
    "concept_drift": "#e34948",
}
REGIME_LABELS = {
    "baseline": "Baseline",
    "data_drift": "Data drift",
    "concept_drift": "Concept drift",
}

# --- Rampas ---------------------------------------------------------------

# Secuencial: un solo tono, claro -> oscuro. Para magnitud.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "churn_seq", ["#cde2fb", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
)

# Divergente: dos tonos opuestos con gris neutro en el centro. Para polaridad
# (correlaciones). Nunca un arcoíris, y nunca un tono en el punto medio.
DIVERGING = LinearSegmentedColormap.from_list(
    "churn_div", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#a52a2a"]
)


def set_style() -> None:
    """Aplica el estilo a matplotlib. Llamar una vez al inicio de cada notebook."""
    mpl.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            # Rejilla y ejes recesivos: el dato es el protagonista.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            # Sin esto, matplotlib dibuja la rejilla ENCIMA de las barras y se
            # ven como líneas claras que las cortan por la mitad.
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


def annotate_regimes(ax, months: list[str], y: float | None = None, alpha: float = 0.07) -> None:
    """Sombrea los tres regímenes del dataset en un eje temporal.

    Aparece en casi todas las figuras del análisis temporal: sin esta referencia
    visual, una línea que sube es solo una línea que sube. Con ella, se ve
    exactamente en qué mes empieza cada fenómeno.
    """
    from src.data.generate_data import regime_for

    spans: list[tuple[int, int, str]] = []
    start = 0
    current = regime_for(0)
    for i in range(1, len(months)):
        regime = regime_for(i)
        if regime != current:
            spans.append((start, i - 1, current))
            start, current = i, regime
    spans.append((start, len(months) - 1, current))

    for first, last, regime in spans:
        ax.axvspan(
            first - 0.5,
            last + 0.5,
            color=REGIME_COLORS[regime],
            alpha=alpha,
            zorder=0,
            linewidth=0,
        )
        ax.annotate(
            REGIME_LABELS[regime],
            xy=((first + last) / 2, y if y is not None else ax.get_ylim()[1]),
            ha="center",
            va="top",
            fontsize=9,
            color=REGIME_COLORS[regime],
            fontweight="semibold",
        )


def save_figure(fig, name: str) -> None:
    """Guarda una figura en docs/img/ para poder embeberla en el README."""
    from src.config import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    print(f"figura guardada: docs/img/{name}.png")


def bare_axis(ax) -> None:
    """Quita la rejilla y el eje Y. Para gráficos con etiquetas directas."""
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelleft=False)


# `plt` se re-exporta a propósito: los notebooks hacen
# `from src.viz import plt, set_style` y así no pueden olvidarse de aplicar el
# estilo por importar matplotlib directamente.
__all__ = [
    "DIVERGING",
    "GRID",
    "INK",
    "INK_MUTED",
    "INK_SECONDARY",
    "REGIME_COLORS",
    "REGIME_LABELS",
    "SEQUENTIAL",
    "SERIES",
    "STATUS",
    "SURFACE",
    "annotate_regimes",
    "bare_axis",
    "plt",
    "save_figure",
    "set_style",
]
