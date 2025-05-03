import numpy as np
import shapely


def extract_point(geometry, index):
    """
    Extract a point from a geometry by index.
    Args:
        geometry (shapely.geometry): The geometry object from which to extract the point.
        index (int): The index of the point to extract.
    Returns:
        shapely.geometry.Point: The extracted point.
    """
    return shapely.geometry.Point(geometry.coords[index])

def extract_nearest(geometry, reference):
    """
    Extract the nearest point from a geometry to a reference point.
    Args:
        geometry (shapely.geometry): The geometry object from which to extract the nearest point.
        reference (shapely.geometry): The reference geometry object to which the nearest point is calculated.
    Returns:
        tuple: A tuple containing the index of the nearest point, the nearest point itself, and the distance to the reference geometry.
    """
    points_geometry = [extract_point(geometry, i) for i in range(len(geometry.coords))]
    distances = [shapely.distance(p, reference) for p in points_geometry]
    index = np.argmin(distances)
    return index, points_geometry[index], distances[index]

def redistribute_vertices(geom, distance):
    """
    Redistribute vertices of a geometry to ensure that the distance between
    consecutive vertices is at most the specified distance (source : https://gis.stackexchange.com/questions/367228/using-shapely-interpolate-to-evenly-re-sample-points-on-a-linestring-geodatafram).
    Args:
        geom (shapely.geometry): The geometry object to redistribute vertices for.
        distance (float): The maximum distance between consecutive vertices.
    Returns:
        shapely.geometry: A new geometry object with redistributed vertices.
    
    """
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return shapely.geometry.LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))