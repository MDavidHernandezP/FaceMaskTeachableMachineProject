import dweepy

thing = input('Inserte nombre de la cosa:')
reading1 = input('Inserte #1 nombre metrico: (Temp) ')
reading2 = input('Inserte #2 nombre metrico: (Humidity) ')
reading3 = input('inserte #3 nombre metrico: (Co2) ')

url = dweepy.get_latest_dweet_for(thing)
dict = url[0]
longdate = dict['created']
date = longdate[:10]
stamptime = longdate[11:19]
tempC = dict['content'][str(reading1)]
hum = dict['content'][str(reading2)]
co2 = dict['content'][str(reading3)]

print("Actualización de estado para",thing)
print("Fecha",date)
print("Tiempo",stamptime)
print("La temperatura actual es...",tempC,"C")
print("La humedad actual es...",hum)
print("El Co2 actual es de...",co2)
