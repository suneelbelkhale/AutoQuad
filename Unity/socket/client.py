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
    if len(sys.argv) != 3:
        print(
            "ERROR. Wrong number of arguments passed. System will exit. Next time please supply 2 arguments!")
        sys.exit()
    else:
        print("2 Arguments exist. We can proceed further")


def checkPort():
    if int(sys.argv[2]) <= 5000:
        print(
            "Port number invalid. Port number should be greater than 5000 else it will not match with Server port. Next time enter valid port.")
        sys.exit()
    else:
        print("Port number accepted!")

checkArg()
try:
    socket.gethostbyname(sys.argv[1])
except socket.error:
    print("Invalid host name. Exiting. Next time enter in proper format.")
    sys.exit()

host = sys.argv[1]
try:
    port = int(sys.argv[2])
except ValueError:
    print("Error. Exiting. Please enter a valid port number.")
    sys.exit()
except IndexError:
    print("Error. Exiting. Please enter a valid port number next time.")
    sys.exit()

checkPort()

host = "127.0.0.1"
port = 6000
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Client socket initialized")
    s.setblocking(0)
    s.settimeout(15)
except socket.error:
    print("Failed to create socket")
    sys.exit()

command = "put test"

CommClient = command.encode('utf-8')
try:
    s.sendto(CommClient, (host, port))
except ConnectionResetError:
    print(
        "Error. Port numbers are not matching. Exiting. Next time please enter same port numbers.")
    sys.exit()
print("Checking for acknowledgement")
try:
    ClientData, clientAddr = s.recvfrom(4096)
except ConnectionResetError:
    print(
        "Error. Port numbers not matching. Exiting. Next time enter same port numbers.")
    sys.exit()
except:
    print("Timeout or some other error")
    sys.exit()

text = ClientData.decode('utf8')
print(text)
print("We shall start sending data.")


c = 0
#Length = len(CL1[1])
data = np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])
size = data.nbytes
sizeS = size#.st_size  # number of packets
#sizeS = sizeS[:-1]
print("File size in bytes: " + str(sizeS))
Num = int(sizeS / 4096)
Num = Num + 1
print("Number of packets to be sent: " + str(Num))
till = str(Num)
tillC = till.encode('utf8')
s.sendto(tillC, clientAddr)
tillIC = int(Num)
GetRun = data.tobytes()
print("LENGTH")
print(len(np.frombuffer(GetRun, dtype=np.uint8)))
i = 0
while tillIC != 0:

    #Run = GetRun.read(1024)
    Run = GetRun[i:i+4096]
    # print(str(Run))
    #CLC = CL[1].encode('utf-8')
    # GetRun.close()
    #propMsg = b"put" + b"|||" + CLC + b"|||" + Run
    s.sendto(Run, clientAddr)
    c += 1
    tillIC -= 1
    print("Packet number:" + str(c))
    print("Data sending in process:")
    i += 4096
# GetRun.close()

print("Sent from Client - Put function")
