import time

import serial
from serial.tools import list_ports

from helpers import DEFAULT_BAUD

# Q-table action index -> one-byte command understood by arduino/goalkeeper/goalkeeper.ino.
# Matches Pong.step: action 0 moves the paddle by -speed, 1 stays, 2 moves by +speed.
LEFT, STAY, RIGHT = b'L', b'S', b'R'
ACTION_COMMANDS = {0: LEFT, 1: STAY, 2: RIGHT}

# Repeat the current command this often; the sketch stops the motor if it hears nothing for 1s
KEEPALIVE_SECONDS = 0.2


def find_arduino_port():
    # Pick the first port that looks like an Arduino (or a common USB-serial adapter)
    for port in list_ports.comports():
        description = f"{port.description} {port.manufacturer or ''}".lower()
        if any(name in description for name in ("arduino", "ch340", "usb serial", "usbmodem")) \
                or "usbmodem" in port.device or "usbserial" in port.device:
            return port.device
    return None


class Arduino:
    def __init__(self, port=None, baud=DEFAULT_BAUD):
        port = port or find_arduino_port()
        if port is None:
            raise RuntimeError("no Arduino found; pass the port explicitly with --port")
        self.serial = serial.Serial(port, baud, timeout=1)
        # Opening the port resets most Arduino boards; wait for the bootloader to finish
        time.sleep(2)
        self.serial.reset_input_buffer()
        self.last_command = None
        self.last_sent_at = 0.0
        print('connected to Arduino on', port)

    def send_action(self, action):
        command = ACTION_COMMANDS[int(action)]
        # The Arduino keeps doing the last command, so only resend on change or as a keepalive
        now = time.monotonic()
        if command != self.last_command or now - self.last_sent_at > KEEPALIVE_SECONDS:
            self.serial.write(command)
            self.last_command = command
            self.last_sent_at = now

    def close(self):
        if self.serial.is_open:
            self.serial.write(STAY)
            self.serial.flush()
            self.serial.close()
