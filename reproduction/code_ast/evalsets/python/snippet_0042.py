import serial #Serial imported for Serial communication
import time #Required to use delay functions
import pyautogui

ArduinoSerial = serial.Serial('COM11',9600) #Create Serial port object called arduinoSerialData
time.sleep(2) #wait for 2 seconds for the communication to get established


#      a(2)       |           up
# e(6) b(3) d(5)  |  left    enter   right
#      c(4)       |          down

while 1:
    incoming = ArduinoSerial.readline() #read the serial data and print it as line    
    #print incoming
    
    if 'b' in incoming:
        pyautogui.typewrite(['space'], 0.2)

    if 'e' in incoming:
        pyautogui.hotkey('left')  

    if 'd' in incoming:
        pyautogui.hotkey('right') 

    if 'a' in incoming:
            pyautogui.hotkey('up')

    if 'c' in incoming:
        pyautogui.hotkey('down')