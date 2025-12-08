#include <ESP32Servo.h>

Servo miServo;

#define servoPin 19 // ESP32 pin GIOP26 connected to servo motor


void setup() {
  miServo.attach(servoPin);  // attaches the servo on ESP32 pin
  miServo.write(0);
}

void loop() {

  delay(2000);
  miServo.write(90);

}
