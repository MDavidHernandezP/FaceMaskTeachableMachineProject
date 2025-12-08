# Implementando funciones por primera vez en la vida.
def resta(a,b):
    return a-b

#---------------
print("Ingrese un valor para a")

a=int(input())
print("Ingrese un valor para b")

b=int(input())
print("respuesta:")

print(resta(a,b))

# Importando pandas por primera vez en la vida.
import pandas as pd

diccionario = {'Ana': [80,95,92],'María': [83,85,98],'Carlos': [75,81,96],'Rodrigo': [65,83,77],'Marcos': [91,89,76]}

D1 = pd.DataFrame(diccionario)
