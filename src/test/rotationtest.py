import numpy as np

la = np.pi/6
lo = np.pi/3

vec = np.matrix([[np.cos(la)*np.cos(lo)],[np.cos(la)*np.sin(lo)],[np.sin(la)]])
matrix1 = np.matrix([[np.cos(-lo), -np.sin(-lo), 0], [np.sin(-lo), np.cos(-lo), 0], [0,0,1]])
matrix2 = np.matrix([[np.cos(-np.pi/2+la),0,np.sin(-np.pi/2+la)],[0,1,0],[-np.sin(-np.pi/2+la),0,np.cos(-np.pi/2+la)]])
print(np.matmul(matrix1,vec))
print(np.matmul(matrix2,np.matmul(matrix1,vec)))