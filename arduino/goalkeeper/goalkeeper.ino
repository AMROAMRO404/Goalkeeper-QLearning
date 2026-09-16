// Goalkeeper motor control.
// Receives one-byte commands from the computer over USB serial:
//   'L' = move left, 'R' = move right, 'S' = stop
// and drives a DC motor through an H-bridge driver (e.g. L298N).

const long BAUD = 9600;

// H-bridge pins; change these to match your wiring
const int IN1 = 7;   // direction pin A
const int IN2 = 8;   // direction pin B
const int ENA = 9;   // speed pin (must be PWM)

const int MOTOR_SPEED = 200;  // 0-255

// Stop the motor if no command arrives for this long (e.g. the computer crashed)
const unsigned long COMMAND_TIMEOUT_MS = 1000;

char currentCommand = 'S';
unsigned long lastCommandTime = 0;

void moveLeft() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, MOTOR_SPEED);
}

void moveRight() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, MOTOR_SPEED);
}

void stopMotor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
}

void applyCommand(char command) {
  switch (command) {
    case 'L': moveLeft(); break;
    case 'R': moveRight(); break;
    case 'S': stopMotor(); break;
    default: return;  // ignore newlines and unknown bytes
  }
  currentCommand = command;
}

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  stopMotor();
  Serial.begin(BAUD);
}

void loop() {
  while (Serial.available() > 0) {
    applyCommand(Serial.read());
    lastCommandTime = millis();
  }

  if (currentCommand != 'S' && millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    applyCommand('S');
  }
}
