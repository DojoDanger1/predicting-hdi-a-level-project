import numpy as np

shapeCoordsAnticlockwise = [
    (0,0,0),
    (1,0,0),
    (2,2,0),
    (0,1,0)
]
shapeCoordsClockwise = shapeCoordsAnticlockwise[::-1]

def findAngle(shape, vertexIndex):
    prevVertex = shape[vertexIndex-1]
    vertex = shape[vertexIndex]
    nextVertex = shape[(0 if vertexIndex == len(shape)-1 else vertexIndex+1)]
    vec1 = (vertex[0]-prevVertex[0], vertex[1]-prevVertex[1], vertex[2]-prevVertex[2])
    vec2 = (nextVertex[0]-vertex[0], nextVertex[1]-vertex[1], nextVertex[2]-vertex[2])
    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]
    det = (vec1[1]*vec2[2]-vec1[2]*vec2[1])-(vec1[2]*vec2[0]-vec1[0]*vec2[2])+(vec1[0]*vec2[1]-vec1[1]*vec2[0])
    return np.arctan2(det, dot)

sum = 0
for i in range(4):
    sum += findAngle(shapeCoordsClockwise, i)
    print(findAngle(shapeCoordsClockwise, i))
print(sum/np.pi*180)
sum = 0
for i in range(4):
    sum += findAngle(shapeCoordsAnticlockwise, i)
    print(findAngle(shapeCoordsAnticlockwise, i))
print(sum/np.pi*180)

