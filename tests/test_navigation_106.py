import numpy as np
import pytest
from PySide6.QtGui import QVector3D

from photomesh.views import OverlayGLView


@pytest.fixture
def view(qtbot):
    widget=OverlayGLView();widget.resize(800,600)
    widget.setCameraPosition(pos=QVector3D(0,0,0),distance=20,elevation=30,azimuth=45)
    qtbot.addWidget(widget)
    return widget


def matrix(view):
    return np.asarray(view.viewMatrix().data()).reshape(4,4).T


def test_full_revolutions_in_both_axes_and_continuous_pole_crossing(view):
    before=matrix(view)
    view.orbit(0,120)
    assert view.opts['elevation']==150
    view.orbit(0,-120)
    view.orbit(360*4,-360*3)
    np.testing.assert_allclose(matrix(view),before,atol=2e-6,rtol=0)
    view.setCameraPosition(elevation=89,azimuth=30)
    a=matrix(view);view.orbit(0,2);b=matrix(view)
    assert view.opts['elevation']==91
    assert np.linalg.norm(a[:3,:3]-b[:3,:3])<.06


@pytest.mark.parametrize('elevation',[90,-90,0,30,120,-140,180])
@pytest.mark.parametrize('relative',['view','view-upright'])
def test_pan_tracks_mouse_in_screen_pixels_at_poles_and_upside_down(view,elevation,relative):
    view.setCameraPosition(elevation=elevation,azimuth=-90)
    viewport=view.getViewport()
    projection=view.projectionMatrix(viewport,viewport)
    def pixel():
        p=(projection*view.viewMatrix()).map(QVector3D(0,0,0))
        return np.array([(p.x()+1)*view.width()/2,(1-p.y())*view.height()/2])
    before=pixel()
    view.pan(40,25,0,relative=relative)
    np.testing.assert_allclose(pixel()-before,[40,25],atol=.03,rtol=0)


def test_global_pan_preserved_and_angle_notification(view,qtbot):
    view.pan(1,2,3)
    assert view.opts['center']==QVector3D(1,2,3)
    with qtbot.waitSignal(view.anglesChanged,timeout=1000):view.orbit(1,1)
