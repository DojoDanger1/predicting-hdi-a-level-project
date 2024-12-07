import numpy as np
import random

EARTH_RADIUS = 6371 # (in km)

# converts angle in degrees to angle in radians
def deg2rad(angle):
    return (np.pi/180)*angle

# finds the distance (in km) along the surface of the earth, between 2 coordinates
def distBetween2Points(p1, p2):
    La1 = deg2rad(p1[0])
    Lo1 = deg2rad(p1[1])
    La2 = deg2rad(p2[0])
    Lo2 = deg2rad(p2[1])
    return EARTH_RADIUS * np.arccos(np.sin(La1)*np.sin(La2) + np.cos(La1)*np.cos(La2)*np.cos(Lo1-Lo2))

# finds the average distance between 2 types of objects, A(x,y)
def averageDistance(x_objects, y_objects, max_x_objects=2000, max_y_objects=2000):
    if len(x_objects) == 0 or len(y_objects) == 0: # the average distance is undefined
        return None
    if len(x_objects) > max_x_objects: # random sample
        x_objects = random.sample(x_objects, max_x_objects)
    # iterate over each x_object, and find the closest y_object
    min_dists = []
    for x_object in x_objects:
        dist_to_ys = []
        if len(y_objects) > max_y_objects:
            y_objects_in_consideration = random.sample(y_objects, max_y_objects)
        else:
            y_objects_in_consideration = y_objects
        for y_object in y_objects_in_consideration:
            dist_to_ys.append(distBetween2Points(x_object, y_object))
        min_dists.append(min(dist_to_ys)) # the closest y_object
    return np.mean(min_dists)

# calculates the area (in km^2) of a given region bound by a set of latitude/longitude coordinates
def calcArea(bounding_coords):
    if bounding_coords[0] == bounding_coords[-1]: # if the first and last coordinates are the same, then delete the last one
        bounding_coords.pop(-1)
    # convert lat/lon pairs to 3d position vectors
    bounding_vectors = []
    for coord in bounding_coords:
        La = deg2rad(coord[0])
        Lo = deg2rad(coord[1])
        bounding_vectors.append(np.matrix([
            [np.cos(La)*np.cos(Lo)],
            [np.cos(La)*np.sin(Lo)],
            [np.sin(La)]
        ]))
    # iterate over each vertex and find its anticlockwise angle
    anticlockwise_angles = []
    for vertex_index, vertex in enumerate(bounding_vectors):
        prev_vertex = bounding_vectors[vertex_index-1]
        next_vertex = bounding_vectors[0 if vertex_index == len(bounding_vectors)-1 else vertex_index+1]
        La = deg2rad(bounding_coords[vertex_index][0])
        Lo = deg2rad(bounding_coords[vertex_index][1])
        # define the rotation and translation matrices
        rotation_matrix_z = np.matrix([
            [np.cos(-Lo), -np.sin(-Lo), 0],
            [np.sin(-Lo), np.cos(-Lo), 0],
            [0, 0, 1]
        ])
        rotation_matrix_y = np.matrix([
            [np.cos(La-np.pi/2), 0, np.sin(La-np.pi/2)],
            [0, 1, 0],
            [-np.sin(La-np.pi/2), 0, np.cos(La-np.pi/2)]
        ])
        translation_vector = np.matrix([
            [0],
            [0],
            [-1]
        ])
        # rotate and translate the plane tangent to the vertex such that it is the xy-plane
        prev_vertex = np.matmul(rotation_matrix_y, np.matmul(rotation_matrix_z, prev_vertex)) + translation_vector
        vertex = np.matmul(rotation_matrix_y, np.matmul(rotation_matrix_z, vertex)) + translation_vector
        next_vertex = np.matmul(rotation_matrix_y, np.matmul(rotation_matrix_z, next_vertex)) + translation_vector
        # project onto the xy-plane
        prev_vertex[2][0] = 0
        next_vertex[2][0] = 0
        # calculate anticlockwise angle between the vectors
        prev_to_current = vertex - prev_vertex
        current_to_next = next_vertex - vertex
        anticlockwise_angle = np.arctan2((prev_to_current.item(1,0)*current_to_next.item(2,0) - prev_to_current.item(2,0)*current_to_next.item(1,0)) - (prev_to_current.item(2,0)*current_to_next.item(0,0) - prev_to_current.item(0,0)*current_to_next.item(2,0)) + (prev_to_current.item(0,0)*current_to_next.item(1,0) - prev_to_current.item(1,0)*current_to_next.item(0,0)), prev_to_current.item(0,0)*current_to_next.item(0,0) + prev_to_current.item(1,0)*current_to_next.item(1,0) + prev_to_current.item(2,0)*current_to_next.item(2,0))
        anticlockwise_angles.append(anticlockwise_angle)
    # find the interior angles from the anticlockwise angles
    sum_anticlockwise_angles = round(sum(anticlockwise_angles), 5)
    if sum_anticlockwise_angles == 0:
        return None
    if sum_anticlockwise_angles < 0:
        anticlockwise_angles = [-1*angle for angle in anticlockwise_angles]
    interior_angles = [np.pi-angle for angle in anticlockwise_angles]
    # return the area
    num_edges = len(bounding_coords)
    return (EARTH_RADIUS ** 2) * (sum(interior_angles) - np.pi*(num_edges-2))

# finds the density of a type of object, D(x)
def density(x_objects, area):
    return len(x_objects)/area