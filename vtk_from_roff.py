from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkExplicitStructuredGrid
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.util import vtkConstants
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import reference, vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkCellLocator,
    vtkExplicitStructuredGrid,
    vtkGenericCell,
    vtkLine,
    vtkPlane,
    vtkPolyData,
    vtkStaticCellLocator,
    vtkUnstructuredGrid,
)
from vtkmodules.vtkFiltersCore import (
    vtkAppendPolyData,
    vtkExplicitStructuredGridCrop,
    vtkExplicitStructuredGridToUnstructuredGrid,
    vtkPlaneCutter,
    vtkClipPolyData,
    vtkUnstructuredGridToExplicitStructuredGrid,
)
from vtkmodules.vtkFiltersGeneral import vtkBoxClipDataSet
from vtkmodules.vtkFiltersGeometry import vtkExplicitStructuredGridSurfaceFilter

import xtgeo
import dash
import webviz_vtk
from webviz_vtk.utils.vtk import b64_encode_numpy


@dataclass
class PropertySpec:
    prop_name: str
    prop_date: Optional[str]


@dataclass
class CellFilter:
    i_min: int
    i_max: int
    j_min: int
    j_max: int
    k_min: int
    k_max: int


@dataclass
class SurfacePolys:
    point_arr: np.ndarray
    poly_arr: np.ndarray


@dataclass
class PropertyScalars:
    value_arr: np.ndarray
    # min_value: float
    # max_value: float


def _create_vtk_esgrid_from_verts_and_conn(
    point_dims: np.ndarray, vertex_arr_np: np.ndarray, conn_arr_np: np.ndarray
) -> vtkExplicitStructuredGrid:

    vertex_arr_np = vertex_arr_np.reshape(-1, 3)
    points_vtkarr = numpy_to_vtk(vertex_arr_np, deep=1)
    vtk_points = vtkPoints()
    vtk_points.SetData(points_vtkarr)

    conn_idarr = numpy_to_vtkIdTypeArray(conn_arr_np, deep=1)
    vtk_cellArray = vtkCellArray()
    vtk_cellArray.SetData(8, conn_idarr)

    vtk_esgrid = vtkExplicitStructuredGrid()
    vtk_esgrid.SetDimensions(point_dims)
    vtk_esgrid.SetPoints(vtk_points)
    vtk_esgrid.SetCells(vtk_cellArray)

    vtk_esgrid.ComputeFacesConnectivityFlagsArray()

    return vtk_esgrid


def xtgeo_grid_to_vtk_explicit_structured_grid(
    xtg_grid: xtgeo.Grid,
) -> vtkExplicitStructuredGrid:

    pt_dims, vertex_arr, conn_arr, inactive_arr = xtg_grid.get_vtk_esg_geometry_data()
    vertex_arr[:, 2] *= -1

    vtk_esgrid = _create_vtk_esgrid_from_verts_and_conn(pt_dims, vertex_arr, conn_arr)

    # Make sure we hide the inactive cells.
    # First we let VTK allocate cell ghost array, then we obtain a numpy view
    # on the array and write to that (we're actually modifying the native VTK array)
    ghost_arr_vtk = vtk_esgrid.AllocateCellGhostArray()
    ghost_arr_np = vtk_to_numpy(ghost_arr_vtk)
    ghost_arr_np[inactive_arr] = vtkDataSetAttributes.HIDDENCELL

    return vtk_esgrid


def _calc_grid_surface(esgrid: vtkExplicitStructuredGrid) -> vtkPolyData:
    surf_filter = vtkExplicitStructuredGridSurfaceFilter()
    surf_filter.SetInputData(esgrid)
    surf_filter.PassThroughCellIdsOn()
    surf_filter.Update()

    polydata: vtkPolyData = surf_filter.GetOutput()
    return polydata


def get_surface(
    grid: vtkExplicitStructuredGrid,
    scalar: np.ndarray,
):

    polydata = _calc_grid_surface(grid)

    # !!!!!!
    # Need to watch out here, think these may go out of scope!
    points_np = vtk_to_numpy(polydata.GetPoints().GetData()).ravel()
    polys_np = vtk_to_numpy(polydata.GetPolys().GetData())
    original_cell_indices_np = vtk_to_numpy(
        polydata.GetCellData().GetAbstractArray("vtkOriginalCellIds")
    )

    surface_polys = SurfacePolys(point_arr=points_np, poly_arr=polys_np)

    mapped_cell_vals = scalar[original_cell_indices_np]
    property_scalars = PropertyScalars(value_arr=mapped_cell_vals)

    return surface_polys, property_scalars


def run_dash(b64_polys, b64_points, b64_scalar, value_range):
    app = dash.Dash()
    app.layout = webviz_vtk.View(
        id="vtk-view",
        style={"height": "90vh"},
        pickingModes=["click"],
        autoResetCamera=True,
        children=[
            webviz_vtk.GeometryRepresentation(
                actor={"scale": (1, 1, 10)},
                # showCubeAxes=True, # Only if scale is 1
                showScalarBar=True,
                colorDataRange=value_range,
                children=[
                    webviz_vtk.PolyData(
                        polys=b64_polys,
                        points=b64_points,
                        children=[
                            webviz_vtk.CellData(
                                [
                                    webviz_vtk.DataArray(
                                        registration="setScalars",
                                        name="scalar",
                                        values=b64_scalar,
                                    )
                                ]
                            )
                        ],
                    )
                ],
                property={"edgeVisibility": True},
            )
        ],
    )

    app.run_server(debug=True)


if __name__ == "__main__":
    # roff_grid_file = "./eclgrid.roff"
    # roff_scalar_file = "./eclgrid--poro.roff"
    roff_grid_file = "./geogrid.roff"
    roff_scalar_file = "./geogrid--phit.roff"
    xtg_grid = xtgeo.grid_from_file(roff_grid_file)
    xtg_scalar = xtgeo.gridproperty_from_file(roff_scalar_file)
    fill_value = np.nan if not xtg_scalar.isdiscrete else -1
    scalar = xtg_scalar.get_npvalues1d(order="F", fill_value=fill_value).ravel()

    esg_grid = xtgeo_grid_to_vtk_explicit_structured_grid(xtg_grid)

    surface_polys, property_scalars = get_surface(esg_grid, scalar)

    polys = surface_polys.poly_arr
    points = surface_polys.point_arr
    scalar = property_scalars.value_arr

    with open("polys.json", "w") as f:
        f.write(json.dumps(polys.tolist()))
    with open("points.json", "w") as f:
        f.write(json.dumps(points.tolist()))
    with open("scalar.json", "w") as f:
        f.write(json.dumps(scalar.tolist()))

    b64_polys = b64_encode_numpy(polys.astype(np.float32))
    b64_points = b64_encode_numpy(points.astype(np.float32))
    b64_scalar = b64_encode_numpy(scalar.astype(np.float32))
    value_range = [
        np.nanmin(scalar),
        np.nanmax(scalar),
    ]

    run_dash(b64_polys, b64_points, b64_scalar, value_range)
