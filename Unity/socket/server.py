'''
Created on Sep 17, 2016

@author: prati
'''
import socket
import time
import os
import sys
import numpy as np

def checkArg():
    """this works omly if executed from command prompt window, it is not designed to work on
    eclipse platform, since by default, len(sys.argv)=1 & therefore the below condition will
    always be displayed as true"""
    if len(sys.argv) != 2:
        print(
            "ERROR. Wrong number of arguments passed. System will exit. Next time please supply 1 argument!")
        sys.exit()
    else:
        print("1 Argument exists. We can proceed further")


def checkPort():
    if int(sys.argv[1]) <= 5000:
        print(
            "Port number invalid. Port number should be greater than 5000. Next time enter valid port.")
        sys.exit()
    else:
        print("Port number accepted!")


def ServerPut():
    print("Sending Acknowledgment of command.")
    msg = "Valid Put command. Let's go ahead "
    msgEn = msg.encode('utf-8')
    s.sendto(msgEn, clientAddr)
    print("Message Sent to Client.")

    print("In Server, Put function")
    if t2[0] == "put":

        BigSAgain = open(t2[1], "wb")
        data = []
        d = 0
        print("Receiving packets will start now if file exists.")
        #print("Timeout is 15 seconds so please wait for timeout at the end.")
        try:
            Count, countaddress = s.recvfrom(4096)  # number of packet
        except ConnectionResetError:
            print(
                "Error. Port numbers not matching. Exiting. Next time enter same port numbers.")
            sys.exit()
        except:
            print("Timeout or some other error")
            sys.exit()

        tillI = Count.decode('utf8')
        tillI = int(tillI)

        #tillI = 100
        #tillI = tillI - 2
        # s.settimeout(2)
        size = None
        while tillI != 0:
            ServerData, serverAddr = s.recvfrom(4096)
            # s.settimeout(2)

            #BigS = open("tmp.txt", "wb")
            temp = np.frombuffer(ServerData, dtype=np.uint8)
            data += list(temp[0::8])
            if size == None:
                size = len(data)
            if(len(data) < size):
                break

            d += 1
            print("Received packet number:" + str(d))

            # tmp.close()
        print(data)


host = ""
checkArg()
try:
    port = int(sys.argv[1])
except ValueError:
    print("Error. Exiting. Please enter a valid port number.")
    sys.exit()
except IndexError:
    print("Error. Exiting. Please enter a valid port number next time.")
    sys.exit()
checkPort()

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Server socket initialized")
    s.bind((host, port))
    print("Successful binding. Waiting for Client now.")
except socket.error:
    print("Failed to create socket")
    sys.exit()

# time.sleep(1)
while True:
    try:
        data, clientAddr = s.recvfrom(4096)
    except ConnectionResetError:
        print(
            "Error. Port numbers not matching. Exiting. Next time enter same port numbers.")
        sys.exit()
    ServerPut()
