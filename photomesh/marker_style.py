"""Shared GCP flag geometry and colors for Photo, Ortho and Model views."""
from PySide6.QtGui import QColor
from .georef import projection_is_pinned, projection_is_refined

COLORS = {"pinned": (40, 220, 60), "placed": (60, 140, 255), "predicted": (190, 190, 190)}

def marker_state(marker):
    if marker.get("disabled", False):
        return "predicted"
    if marker.get("locked", False) or any(projection_is_pinned(v) for v in marker.get("projections", {}).values()):
        return "pinned"
    return "placed" if any(projection_is_refined(v) for v in marker.get("projections", {}).values()) else "predicted"

def color(state):
    return QColor(*COLORS.get(state, COLORS["predicted"]))

HOVER_COLOR = QColor(255, 65, 59)


def flag_at(x, y, anchors):
    """Resolve overlapping flags using the same screen geometry for hover and clicks."""
    nearest = None
    for label, sx, sy in anchors:
        dx, dy = x - sx, y - sy
        if -5 <= dx <= 18 and -27 <= dy <= 6:
            hit = (dx * dx + dy * dy, label)
            if nearest is None or hit < nearest:
                nearest = hit
    return nearest[1] if nearest is not None else None


def draw_flag(painter, x, y, colour, active=False):
    """The bottom point is the exact GCP; the flag is offset above it."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QPen, QBrush, QPolygonF, QPainter
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    pole = [(x, y), (x, y - 23)]
    for col, width in ((QColor(0, 0, 0, 210), 4.5 if active else 3.5),
                       (colour, 2.5 if active else 1.8)):
        painter.setPen(QPen(col, width))
        painter.drawLine(QPointF(*pole[0]), QPointF(*pole[1]))
    painter.setPen(QPen(QColor(0, 0, 0, 210), 1.2))
    painter.setBrush(QBrush(colour))
    painter.drawPolygon(QPolygonF([QPointF(x, y - 23), QPointF(x + 13, y - 19),
                                  QPointF(x, y - 14)]))
    painter.setPen(QPen(QColor(0, 0, 0), 1.3))
    painter.drawEllipse(QPointF(x, y), 2.8, 2.8)
    painter.restore()


def flag_icon(state, size=24):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap, QIcon, QPainter
    pm = QPixmap(size, size); pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.scale(size / 28, size / 28)
    try:
        draw_flag(painter, 6, 26, color(state))
    finally:
        painter.end()
    return QIcon(pm)
