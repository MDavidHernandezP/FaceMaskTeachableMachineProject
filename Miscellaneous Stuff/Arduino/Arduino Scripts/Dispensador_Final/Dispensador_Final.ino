#include <WiFi.h>
#include <WifiUDP.h>
#include <NTPClient.h>
#include <Time.h>
#include <TimeLib.h>
#include <Timezone.h>
#include <ESP32Servo.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>

Servo servo;
int Comando_entrada;
    int PINSERVO = 13;
    int PULSOMIN = 900;
    int PULSOMAX = 2000;

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// Configurar wifi
const char* ssid = "IZZI-C06C";                                                                                                
const char* password = "2C9569AFC06C";                                                                                                                                                                          

// Definir propiedades NTP
#define NTP_OFFSET   60 * 60                                                                                               
#define NTP_INTERVAL 60 * 1000                                                                                             
#define NTP_ADDRESS  "pool.ntp.org"                                                                                        
WiFiUDP ntpUDP;                                                                                                           
NTPClient timeClient(ntpUDP, NTP_ADDRESS, NTP_OFFSET, NTP_INTERVAL);
TimeChangeRule CDT = {"CDT", Second, Sun, Mar, 2, -360};                                                                     
TimeChangeRule CST = {"CST", First, Sun, Nov, 2, -420};                                                                       
Timezone CT(CDT, CST);
time_t local, utc;

const char * days[] = {"Domingo", "Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado"} ;                        
const char * months[] = {"Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"} ;            


void setup() 
{
  servo.attach(PINSERVO, PULSOMIN, PULSOMAX);
  Serial.begin(9600);                                                                                                    
  Serial.println("");
  Serial.print("conectando a ");                                                                                          
  Serial.print(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)                                                                                    
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Conectando WiFi a ");                                                                                   
  Serial.print(WiFi.localIP());                                                          
  Serial.println("");

  // put your setup code here, to run once:
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();

}

void loop() 
{
  if (WiFi.status() == WL_CONNECTED)                                                                                    
  {   
    timeClient.update();                                                                                                
    unsigned long utc =  timeClient.getEpochTime();
    local = CT.toLocal(utc);                                                                                           
    printTime(local);                                                                                                   
  }
  else {                                                                                                               
    WiFi.begin(ssid, password);
    delay(1000);
  }
  delay(1000);    // Enviar una solicitud para actualizar cada 10 seg (= 10,000 ms)
}

void printTime(time_t t)                                                                                             
{

  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10,10);
  display.println("Fecha local: ");
  display.display();
  
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10,20);
  display.print(convertirTimeATextoFecha(t));
  display.display();
  
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10,30);
  display.print(convertirTimeATextoHora(t));
  display.display();
  display.clearDisplay();
  delay(10000);
}

String convertirTimeATextoFecha(time_t t)                                                                              
{
  String date = "";
  date += days[weekday(t)-1];
  date += ", ";
  date += day(t);
  date += " ";
  date += months[month(t)-1];
  date += ", ";
  date += year(t);
  return date;
}

String convertirTimeATextoFechaSinSemana(time_t t)                                                                   
{
  String date = "";
  date += months[month(t)-1];
  date += "   ";
  date += year(t);
  return date;
}

String convertirTimeATextoHora(time_t t)                                                                                                                                                    
{ 
  if(hour(t) == 20 && minute(t) == 55){   //Primera comida 
    servo.write(180);
    delay(2000);
    servo.write(0);
    delay (60000);
  } 
  if (hour(t) == 20 && minute(t) == 56){   //Segunda comida 
    servo.write(180);
    delay(2000);
    servo.write(0);
    delay (60000);
  }
  if (hour(t) == 20 && minute(t) == 57){   //Tercera comida 
    servo.write(180);
    delay(2000);
    servo.write(0);
    delay (60000);
  } 
  
  String hora ="";                                                                                                    
  if(hour(t) < 10)
  hora += "0";
  hora += hour(t);
  hora += ":";
  if(minute(t) < 10)                                                                                                  
    hora += "0";
  hora += minute(t);
  return hora;
  
} 
