import socket
import struct
import pandas as pd
import numpy as np
import time

df = pd.read_csv('normal-0927-afterclean.csv')
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ndf = np.array(df).tolist()
count = 0
# print(df)

for data in ndf:
    sdata = struct.pack('29i', data[1],data[2],data[3],data[4],data[5],data[6],0,data[8],data[9],data[10],data[11],data[12],
                        data[13],data[14],data[15],data[16],data[17],data[18],data[19],data[20],data[21],data[22],data[23],data[24],data[25],
                        data[26],data[27],data[28],data[29])
    s.sendto(sdata,('10.108.187.23', 9898))
    count+=1
    if count == 8:
        time.sleep(0.005)
        count = 0

s.close()

