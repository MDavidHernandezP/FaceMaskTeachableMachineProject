#Se define la multiplicación
def Área(pi,r):
    return (math.pi)*(r*r)

#Importa la librería math
import math

#Imprime el valor de pi
print("El valor de pi es:")

print(math.pi)

print("Escribe el valor del radio(r) de un circulo")
r=int(input())

#Resultado
print(Área(math.pi,r))
