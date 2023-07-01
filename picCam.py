from picamera2 import Picamera2
from time import sleep
import datetime
import logging
import sys

#pengaturan kamera
picam2 = Picamera2()
camConfig = picam2.create_still_configuration(main={"size" : (2592, 1944)}, lores={"size": (640, 480)})
picam2.configure(camConfig)
picam2.set_controls({"ExposureTime": 70000, "AnalogueGain": 8.0})
picam2.start()

def logger(path):
  # mencatat log pada program
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.DEBUG)
  stdout_handler.setFormatter(formatter)
  file_handler = logging.FileHandler(f'{path}logs.log')
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  logger.addHandler(stdout_handler)

def takePic(path):
  # mengambil foto dan memberi nama sesuai tanggal dan waktu
  date = datetime.datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
  namePic = f"telor-{date}.jpg"
  picam2.capture_file(f"{path}/{namePic}")
  return namePic