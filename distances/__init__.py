import numpy as np
import pandas as pd
import shapely

############################
# Polygon method
############################

def distance_abs(front1, front2):
    '''
    Calculates the absolute distance between two fronts, by dividing the area of the polygon by the mean length.
    If the front lengths are too different (50%), returns None.
    '''
    if np.abs(front1.length - front2.length)/np.min([front1.length, front2.length]) >= 0.5:
        return None
    else:
        points_1 = [shapely.Point(coords) for coords in front1.coords]
        points_2 = [shapely.Point(coords) for coords in front2.coords]

        if shapely.distance(points_1[0], points_2[0]) > shapely.distance(points_1[0], points_2[-1]):
            points_2 = points_2[::-1]
            front2 = shapely.LineString(points_2)
            
        polygon = shapely.Polygon(list(front1.coords) + list(front2.coords[::-1]))
        area = polygon.area
        d = area / ((front1.length + front2.length) / 2)
        return 2 * area / (front1.length + front2.length)
    
def distance_sgn(front1, front2, u):
    points_interp_1 = pd.DataFrame(
        [],
        columns=["points", "x", "y"]
    )
    points_interp_1.loc[:, "points"] = front1.interpolate(np.linspace(0, front1.length, 100))
    points_interp_1.loc[:, ["x", "y"]] = np.stack([np.array([points_interp_1.points.iloc[i].x, points_interp_1.points.iloc[i].y]) for i in range(100)], axis=0)
    ll = shapely.MultiLineString([np.stack([points_interp_1.loc[:, ["x", "y"]].iloc[i] - u * 50000, points_interp_1.loc[:, ["x", "y"]].iloc[i] + u * 50000])for i in range(100)])
    points_interp_2 = pd.DataFrame(
        [],
        columns=["points", "x", "y"]
    )
    for i in range(len(points_interp_1)):
        l = ll.geoms[i]
        point_interp_2_i = shapely.intersection(front2, l)
        
        if point_interp_2_i.is_empty:
            points_interp_2.loc[len(points_interp_2), ["points", "x", "y"]] = [None, None, None]
            
        elif isinstance(point_interp_2_i, shapely.Point):
            points_interp_2.loc[len(points_interp_2), ["points", "x", "y"]] = [point_interp_2_i, point_interp_2_i.x, point_interp_2_i.y]
        
        elif isinstance(point_interp_2_i, shapely.MultiPoint):
            point_interp_2_i = np.array([np.array([pt.x, pt.y]) for pt in point_interp_2_i.geoms]).mean(axis=0)
            point_interp_2_i = shapely.Point(point_interp_2_i)
            points_interp_2.loc[len(points_interp_2), ["points", "x", "y"]] = [point_interp_2_i, point_interp_2_i.x, point_interp_2_i.y]

    points_interp = pd.concat([points_interp_1, points_interp_2], axis=1)
    points_interp.columns = ["points_1", "x1", "y1", "points_2", "x2", "y2"]
    points_interp.dropna(inplace=True)
    
    if len(points_interp) == 0:
        return None
    else:
        dot_prod = points_interp.loc[:, ["x1", "y1", "x2", "y2"]].apply(lambda z: (z.x2 - z.x1) * u[0] + (z.y2 - z.y1) * u[1], axis=1)
        sgn = np.sign(dot_prod.mean())
        return sgn

def distance(front1, front2, dir_1, dir_2):
    assert dir_1 == dir_2, "Fronts must have the same direction" # not necessary for the calculation, just sanitary check.
    dist_abs = distance_abs(front1, front2)
    if dist_abs is None:
        return None, 1
    else:
        # transform direction into av vector:
        if dir_1 == "N":
            u = np.array([0, 1])
        elif dir_1 == "S":
            u = np.array([0, -1])
        elif dir_1 == "E":
            u = np.array([1, 0])
        elif dir_1 == "W":
            u = np.array([-1, 0])
        elif dir_1 == "NE":
            u = np.array([1, 1]) / np.sqrt(2)
        elif dir_1 == "NW":
            u = np.array([-1, 1]) / np.sqrt(2)
        elif dir_1 == "SE":
            u = np.array([1, -1]) / np.sqrt(2)
        elif dir_1 == "SW":
            u = np.array([-1, -1]) / np.sqrt(2)
        else:
            raise ValueError(f"Unknown direction: {dir_1}")
        
        sgn = distance_sgn(front1, front2, u)
        
        if sgn is None:
            return None, 2
        else:       
            return sgn * dist_abs, 0
        
        
############################
# Box method
############################
        
def box_distance(front1, front2, box, dir_1, dir_2):
    '''
    Implements the box distance between two fronts, given the box and the direction of the fronts.
    Returns the distance and an error code:
    - 0 : success
    - 1 : fronts do not intersect twice with the box sides.
    '''
    # Sanity check : fronts must have the same direction.
    assert dir_1 == dir_2, "Fronts must have the same direction"
    
    # transform direction into av vector:
    if dir_1 == "N":
        u = np.array([0, 1])
    elif dir_1 == "S":
        u = np.array([0, -1])
    elif dir_1 == "E":
        u = np.array([1, 0])
    elif dir_1 == "W":
        u = np.array([-1, 0])
    elif dir_1 == "NE":
        u = np.array([1, 1]) / np.sqrt(2)
    elif dir_1 == "NW":
        u = np.array([-1, 1]) / np.sqrt(2)
    elif dir_1 == "SE":
        u = np.array([1, -1]) / np.sqrt(2)
    elif dir_1 == "SW":
        u = np.array([-1, -1]) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown direction: {dir_1}")
    
    inter1, inter2 = shapely.intersection(front1, box.exterior), shapely.intersection(front2, box.exterior)
    if isinstance(inter1, shapely.Point):
        inter1 = shapely.MultiPoint([inter1])
    if isinstance(inter2, shapely.Point):
        inter2 = shapely.MultiPoint([inter2])
    
    if inter1.is_empty or inter2.is_empty:
        return None, 1    
    elif (len(list(inter1.geoms)) < 2) or (len(list(inter2.geoms)) < 2):
        return None, 1
    else:
        # find the base of the box:
        # 1 - separate the box into sides
        box_sides = [shapely.LineString([box.exterior.coords[i], box.exterior.coords[i + 1]]) for i in range(len(box.exterior.coords) - 1)]
        # 2 - exclude sides that intersect with the fronts and calculate the centers of the others 
        possible_base = [box_sides[i] for i in range(len(box_sides)) if not shapely.intersects(box_sides[i], front1)]
        possible_base_centers = np.array([[shapely.centroid(possible_base[i]).x, shapely.centroid(possible_base[i]).y] for i in range(len(possible_base))])
        # 3 - calculate the dot product of the centroid of the sides with the vector and minimize it to find the base
        u_dot_c = np.sum(possible_base_centers * np.stack([u, u], axis=0), axis=1)
        base = possible_base[np.argmin(u_dot_c)]

        # cropped fronts
        front1_crop = shapely.intersection(front1, box)
        front2_crop = shapely.intersection(front2, box)
        
        if isinstance(front1_crop, shapely.MultiLineString):
            lines = [shapely.LineString(line) for line in front1_crop.geoms]
            front1_crop = lines[np.argmax([line.length for line in lines])]
        if isinstance(front2_crop, shapely.MultiLineString):
            lines = [shapely.LineString(line) for line in front2_crop.geoms]
            front2_crop = lines[np.argmax([line.length for line in lines])]

        # distance
        points_1 = [shapely.Point(coords) for coords in front1_crop.coords]
        points_2 = [shapely.Point(coords) for coords in front2_crop.coords]
        base_points = [shapely.Point(coords) for coords in base.coords]
        # change the order of the points if necessary to avoid negative area
        if shapely.distance(points_1[0], base_points[0]) > shapely.distance(points_1[0], base_points[-1]):
            points_1 = points_1[::-1]
            front1_crop = shapely.LineString(points_1)
            
        if shapely.distance(points_2[0], base_points[0]) > shapely.distance(points_2[0], base_points[-1]):
            points_2 = points_2[::-1]
            front2_crop = shapely.LineString(points_2)
            
        polygon1 = shapely.Polygon(list(front1_crop.coords) + list(base.coords[::-1]))
        polygon2 = shapely.Polygon(list(front2_crop.coords) + list(base.coords[::-1]))
        area1 = polygon1.area
        area2 = polygon2.area
        return (area2 - area1)/base.length, 0
    
############################
# Curvilinear centerline method
############################

def curvilinear_distance(front1, front2, cl, dir_1, dir_2):
    '''
    Implements the curvilinear distance between two fronts, given the curvilinear centerline and the direction of the fronts.
    Returns the distance and an error code:
        - 0 : success
        - 1 : fronts do not intersect with the centerline.
    '''
    # Sanity check : fronts must have the same direction.
    assert dir_1 == dir_2, "Fronts must have the same direction"
    
    # transform direction into av vector:
    if dir_1 == "N":
        u = np.array([0, 1])
    elif dir_1 == "S":
        u = np.array([0, -1])
    elif dir_1 == "E":
        u = np.array([1, 0])
    elif dir_1 == "W":
        u = np.array([-1, 0])
    elif dir_1 == "NE":
        u = np.array([1, 1]) / np.sqrt(2)
    elif dir_1 == "NW":
        u = np.array([-1, 1]) / np.sqrt(2)
    elif dir_1 == "SE":
        u = np.array([1, -1]) / np.sqrt(2)
    elif dir_1 == "SW":
        u = np.array([-1, -1]) / np.sqrt(2)
    else:
        raise ValueError(f"Unknown direction: {dir_1}")

    if not (shapely.intersects(front1, cl) and shapely.intersects(front2, cl)):
        return None, 1
    
    else:
    
        points_cl = shapely.get_coordinates(cl)

        ## orient the centerline such that the the origin is upflow
        if np.dot(points_cl[-1, :], u) < np.dot(points_cl[0], u):
            cl = cl.reverse()
            points_cl = shapely.get_coordinates(cl)

        ## define curvilinear abscissa
        x = np.array([0] + [shapely.LineString(points_cl[0:i + 1, :]).length for i in range(1, points_cl.shape[0])]) # curvilinear abscissa of all the points along the centerline

        ## weights
        p = 2
        points1 = shapely.get_coordinates(front1)
        cl1_mesh_0, f1_mesh_0 = np.meshgrid(points_cl[:, 0], points1[:, 0], indexing="ij")
        cl1_mesh_1, f1_mesh_1 = np.meshgrid(points_cl[:, 1], points1[:, 1], indexing="ij")
        cl1_mesh, f1_mesh = np.stack([cl1_mesh_0, cl1_mesh_1], axis=-1), np.stack([f1_mesh_0, f1_mesh_1], axis=-1)
        w1 = 1 / np.sum(np.abs(cl1_mesh - f1_mesh)** p, axis=-1) ** (1/p)
        w1 = w1 / np.sum(w1, axis=0)  # normalize weights

        x1_mesh = np.stack([x for _ in range(points1.shape[0])], axis=1)
        t1 = np.sum(x1_mesh * w1, axis=0)
        T1 = t1.mean()

        points2 = shapely.get_coordinates(front2)
        cl2_mesh_0, f2_mesh_0 = np.meshgrid(points_cl[:, 0], points2[:, 0], indexing="ij")
        cl2_mesh_1, f2_mesh_1 = np.meshgrid(points_cl[:, 1], points2[:, 1], indexing="ij")
        cl2_mesh, f2_mesh = np.stack([cl2_mesh_0, cl2_mesh_1], axis=-1), np.stack([f2_mesh_0, f2_mesh_1], axis=-1)
        w2 = 1 / np.sum(np.abs(cl2_mesh - f2_mesh)** p, axis=-1) ** (1/p)
        w2 = w2 / np.sum(w2, axis=0)  # normalize weights

        x2_mesh = np.stack([x for _ in range(points2.shape[0])], axis=1)
        t2 = np.sum(x2_mesh * w2, axis=0)
        T2 = t2.mean()
        
        return T2 - T1, 0