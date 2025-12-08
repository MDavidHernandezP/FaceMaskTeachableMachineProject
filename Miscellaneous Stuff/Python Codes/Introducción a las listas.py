"""
Introducción a las listas
"""

#Crear una lista
L = ['Elemento 1', 'Elemento 2', 'Elemento 3', 'Elemento 4', 'Elemento 5']
# Imprimir la lista
print(L)

"""
Indexación
"""

#Imprimir el elemento 1 de la lista
elemento_1 = L[0]
print('El primer elemento es:',elemento_1)

#Imprimir el elemento 3 de la lista
elemento_3 = L[2]
print('El tercer elemento es:',elemento_3)

#Los elementos de una lista se pueden modificar. Para ello es necesario especificar el índice correspondiente y asignarle un nuevo valor.
print('Lista original:',L)
#Modificar el primer elemento de la Lista
L[0] = 'Ana'
#Imprimir la lista
print('Nueva lista:',L)

#Edita el código para modificar el segundo elemento por el número 17.
L[1] = 'setzo'
print('Nueva lista 2:',L)

#Cambiando todos los elementos de la lista :)
L[0] = 'Copulación'
L[1] = 'Sexo sin condón'
L[2] = 'Extasis'
L[3] = 'cum'
L[4] = 'Bendición'
print('Nueva Lista 3:',L)

#Si tratas de utilizar un índice que no está en la lista maracará error === L[5] ===

#También se puede obtener una parte de una lista especificando el índice inicial y el índice final.
L2 = [10,15,5,2,20,8]
#Imprimir del elemento 3 al 6
print(L[2:6])

#Ejemplo sexi hecho por mi :)
L3 = [340,353,53535,134124,4,21424,24142,24124,]
print(L[3:6])

#Para conocer el tamaño de una lista se puede utilizar la función len(), la cual devuelve la cantidad de objetos que contiene.
#Imprimir el tamaño de la lista
print(len(L))

"""
Iterar listas
"""

#Se puede acceder a cada elemento de una lista utilizando el ciclo for.
Lista = [10,2,8,9]
for l in Lista:
  print(l)

#Para obtener el índice del elemento se puede utilizar la función enumerate():
for i,l in enumerate(Lista):
  print(i,l)

#También se puede iterar utilizando len()
for i in range(0, len(Lista)):
    print(Lista[i])

#Ejercicio: Crea una lista con los números enteros del 1 al 10. Calcula el cubo de cada elemento y sustituye el resultado en la misma lista. Al final, imprime la lista
L = [1,2,3,4,5,6,7,8,9,10]
print(L)
print('Al cuadrado')
L[0] = 1
L[1] = 4
L[2] = 9
L[3] = 16
L[4] = 25
L[5] = 36
L[6] = 49
L[7] = 64
L[8] = 81
L[9] = 100
print('Nueva lista:',L)

Lista = [1,2,3,4,5,6,7,8,9,10]
Lista[0] = 1
Lista[1] = 4
Lista[2] = 9
Lista[3] = 16
Lista[4] = 25
Lista[5] = 36
Lista[6] = 49
Lista[7] = 64
Lista[8] = 81
Lista[9] = 100
for l in Lista:
  print(l)

"""
Métodos
"""

"""
#Las listas integran unos métodos que se pueden utilizar para realizar algunas acciones. A continuación se enlistan algunos de ellos:
#Crear la lista
L = [10,15,5,2,20,8]

#1. append(): Agrega un elemento al final de la lista.

#Utilizar append() para agregar un elemento al final de la lista
L.append(1)
print(L)

#2. count(): Recibe un elemento y devuelve un número correspondiente a la cantidad de veces que aparece ese elemento en la lista.

#Utilizar count() para conocer la cantidad de veces que aparece el número 15 en la lista
n = L.count(15)
print(n)

#3. index(): Recibe un elemento devuelve el índice de la posición del elemento en la lista.

#Utilizar index() para conocer la posición del número 5 en la lista
i = L.index(5)
print(i)

#4. insert(): Agrega un elemento en la posición indicada

#Utilizar insert() para añadir el número 11 en la posición 4
L.insert(4,11)
print(L)

#5. sort(): Ordena los elementos de la lista

#Utilizar sort() para ordenar los elementos de la lista
#Orden ascendente
L.sort()
print(L)
#Orden descendente
L.sort(reverse=True)
print(L)

#6. copy(): Devuelve una nueva lista que contiene una copia de la original.

#Copiar una lista
copia = L[:]
print(copia)
#Copiar una lista con el método copy()
copia = L.copy()
print(copia)

#Ejercicio:
#a) Crea una lista con los nombres de los colores amarillo, azul y rojo.
#b) Determina el índice del color rojo y utiliza ese índice para agregar el color morado antes del rojo.
#c) Agrega el color verde al final de la lista.
#d) Ordena los elementos en orden descendente.

L = ['amarillo','azul','rojo']
L.insert(3,'morado')
print(L)
L.append('verde')
print(L)
L.sort(reverse=True)
print(L)
"""

"""
Listas de dos dimensiones
"""



#Podemos acceder a cada uno de los elementos a través de la fila y la columna de la siguiente manera:
#L[i][j] donde i es la fila y j la columna.

L = [[10,8,7,10],[8,9,5,10],[5,6,8,7]]
#Imprimir el valor de la fila 0 y columna 2  
valor = L[0][2]
print(valor)

#Para acceder uno a uno a los elementos de la lista, se puede utilizar dos ciclos for anidados:

#Imprimir los elementos de la lista
#El primer ciclo for obtiene las filas, una a la vez.
#En cada iteración del primer ciclo for, el for interno obtiene el elemento de cada columna de la fila.  
for fila in L:
  for num in fila:
    #Imprime el elemento con un espacio
    print(num, end=' ')
  #Imprime salto de línea
  print()



"""
Ejercicio
"""

L2 = [[10, 7, 3], [20, 4, 17]]
print('Lista original',L2)
L2 = L2[0][0] + L2[0][1] + L2[0][2] + L2[1][0] + L2[1][1] + L2[1][2]
print('Nueva lista',L2)
L2 = L2/6
print('Nueva lista final',L2)



#Escribe aquí tu código
L2 = [[10, 7, 3], [20, 4, 17]]
for fila in L2:
  for num in fila:
    #Imprime el elemento con un espacio
    print(num, end=' ')
  #Imprime salto de línea
  print()
L2 = L2[0][0] + L2[0][1] + L2[0][2] + L2[1][0] + L2[1][1] + L2[1][2]
print('Sumatoria',L2)
L2 = L2/6
print('Promedio',L2)
