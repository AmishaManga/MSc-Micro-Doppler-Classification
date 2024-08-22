# The purpose of this script is to log data from the LCR and 
# To a create a small text file with a desciption on what data is being logged.

import zmq
import datetime

socket = zmq.Context().socket(zmq.SUB)
socket.connect ("tcp://%s:%s" % ('192.168.0.10',5557))
topicfilter = ""
socket.setsockopt_string(zmq.SUBSCRIBE, topicfilter)

f = open('{:%Y-%m-%d-%H-%M-%S}.log'.format(datetime.datetime.now()), "wb")

isData = True

while True:
   msg = socket.recv()
   f.write(msg)
   if msg is None:
       isData = False
       socket.close()
       f.close()
       print("no data")

    
    
  