const int sensor1 = 6; // pre-reward gate
const int sensor2 = 7; // post reward gate
int state = LOW;
int val = 0;

void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  // PIR motion sensor is determined is an input here.  
  pinMode(sensor1, INPUT); 
  pinMode(sensor2, INPUT); 
  Serial.begin(9600);   
}

void loop() {
  delay(50); // 50ms
  if (state == LOW) {
    val = digitalRead(sensor1);
    if (val == HIGH) {           
      state = HIGH;
      Serial.println("1");
      // turn off LED when pre-gate detected
      digitalWrite(LED_BUILTIN, LOW);
      // Serial.println("Motion detected"); 
    }
  }
  else {
    val = digitalRead(sensor2);
    Serial.println("2"); 
    if (val == HIGH) {           
      // dispense reward!
      digitalWrite(LED_BUILTIN, HIGH);
      Serial.println("3"); 
      state = LOW;
      // prevent long trigger of sensor1 from resetting immediately
      delay(3000);
    }
  }  
}