from luma.core.interface.serial import i2c, spi
from luma.oled.device import sh1106
from luma.core.render import canvas
from PIL import Image
from picCam import *
from process import *
import Adafruit_GPIO.SPI as SPI
import RPi.GPIO as GPIO
import os 

#atur Display OLED
serial = i2c(port=1, address=0x3C)
device = sh1106(serial)

# Clear display.
device.clear()

#tombol
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
tomUp = 27 #atas
tomDo = 10 #abawah
tomOk = 17 #OK
tomBa = 11 #kembali
GPIO.setup(tomUp,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(tomDo,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(tomOk,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(tomBa,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)

#LED
ledRR = 5 #LED kanan merah
ledRG = 6 #LED kanan hijau
ledLR = 13 #LED kiri merah
ledLG = 19 #LED kiri hijau
hpLed = 22 #HPL
GPIO.setup(ledRR,GPIO.OUT)
GPIO.setup(ledRG,GPIO.OUT)
GPIO.setup(ledLR,GPIO.OUT)
GPIO.setup(ledLG,GPIO.OUT)
GPIO.setup(hpLed,GPIO.OUT)
GPIO.output(ledRR,False)
GPIO.output(ledRG,False)
GPIO.output(ledLR,False)
GPIO.output(ledLG,False)
GPIO.output(hpLed,False)  

#fungsi Tampilan Menu
menuLok = 0
menuPre = 0
menuNow = menuPre
tekanArah = False
tekanOkBa = False
homeDir = "/home/pitik/Progres/"

def showMenu(idMen,lok):
  if lok == 0:#lokasi 0 menu awal
    path = f"{homeDir}display/menu/"
  elif lok == 1:#lokasi 1 menu opsi
    path = f"{homeDir}display/opsi/"
  elif lok == 2:#lokasi menu daya
    path = f"{homeDir}display/daya/"
  elif lok == 3:#lokasi riwayat hasil
    path = f"{homeDir}HasilPRD/"
  global listMenu
  listMenu = os.listdir(path)
  if len(listMenu) != 0:
    if lok != 3:
      listMenu.sort()
    else:
      listMenu.sort(reverse=True)
    menuImg = Image.open(f"{path}{listMenu[idMen]}").convert('1')
    with canvas(device) as draw:
      draw.bitmap((0, 0), menuImg, fill="white")
      if lok == 3:
        draw.text((0, 0), text=f"{idMen+1}", fill="white")
  else:
    device.display(Image.open(f"{homeDir}display/rwytKsng.png").convert('1'))

  # device.display(menuImg)
  
#Kombinasi Hasil
def hasilPro(hasKi,hasKa,namePic):
  img1 = Image.open(f"{homeDir}display/hasil/kiri/{hasKi}.png").convert('1')
  img2 = Image.open(f"{homeDir}display/hasil/kanan/{hasKa}.png").convert('1')
  mask = Image.open(f"{homeDir}display/hasil/mask.png").convert('1')
  print(f"{homeDir}display/hasil/kiri/{hasKi}.png")
  print(f"{homeDir}display/hasil/kanan/{hasKa}.png")
  hasil = Image.composite(img1, img2, mask)
  hasil.save(f"{homeDir}HasilPRD/{namePic}")
  while not GPIO.input(tomBa):
    device.display(hasil)
    if(hasKi != 5):#nilai 5 berarti infertil selain itu fertil
      GPIO.output(ledLG,True)
      GPIO.output(ledLR,False)
    else:
      GPIO.output(ledLG,False)
      GPIO.output(ledLR,True)
    if(hasKa != 5):#nilai 5 berarti infertil selain itu fertil
      GPIO.output(ledRG,True)
      GPIO.output(ledRR,False)
    else:
      GPIO.output(ledRG,False)
      GPIO.output(ledRR,True)
  # dimatikan lagi
  GPIO.output(ledRR,False)
  GPIO.output(ledRG,False)
  GPIO.output(ledLR,False)
  GPIO.output(ledLG,False)
  GPIO.output(hpLed,False)

#fungsi Shutdown sistem
def shut_down():
    print("shutting down")
    command = "/usr/bin/sudo /sbin/shutdown -h now"
    device.clear()
    import subprocess
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
    print(output)

# Riwayat
def riwayatHps():
  print("Menghapus Data")
  file = open(f"{homeDir}log/hasil.log",'w')
  file.close()
  file = open(f"{homeDir}log/logs.log",'w')
  file.close()
  path = f"{homeDir}hasilCam/"
  for picName in os.listdir(path):
    picFile = path+picName
    os.remove(picFile)
  path = f"{homeDir}HasilPRD/"
  for picName in os.listdir(path):
    picFile = path+picName
    os.remove(picFile)
  while not GPIO.input(tomBa):
    device.display(Image.open(f"{homeDir}display/rwytKsng.png").convert('1'))

# memulai logging 
logger(f"{homeDir}log/")

#LOOP
while True:
  showMenu(menuNow,menuLok) 
  #bagian tekan arah
  if not tekanArah:
    if GPIO.input(tomDo):
      tekanArah = True
      #menekan arah menambah nilai menuNow menunjukkan posisi text yang dipilih
      menuNow +=1
      if  menuNow > len(listMenu)-1:
        menuNow = 0
      menuPre = menuNow
      print("bawah")
      print(f"menuNow = {menuNow}")
      print(f"menuLok = {menuLok}")
    elif GPIO.input(tomUp):
      tekanArah = True
      menuNow -= 1
      if  menuNow < 0:
        menuNow = len(listMenu)-1
      menuPre = menuNow
      print("atas")
      print(f"menuNow = {menuNow}")
      print(f"menuLok = {menuLok}")
  #mencegah arah tertekan terus
  elif not GPIO.input(tomUp) and not GPIO.input(tomDo):
    tekanArah = False
  
  #ketika menekan tombol ok atau back
  if not tekanOkBa:
    if GPIO.input(tomOk):
      tekanOkBa = True
      #perubahan posisi menu
      #menuLok0 menu paling depan
      if menuLok == 0:
        if menuNow == 0:
          #menu di cek telur
          GPIO.output(hpLed,True)
          print("cek telur")
          # loading()
          imgLoad= Image.open(f"{homeDir}display/loading.png").convert('1')
          device.display(imgLoad)
          # phoSel = img2 = Image.open(f"{homeDir}display/photo.png").convert('1')
          fName = takePic(f"{homeDir}hasilCam/")
          hCek = cekPic(fName)
          hasilPro(hCek[0],hCek[1],fName)
        elif menuNow == 1:
          # menulok1 = menu opsi
          menuLok = 1
          menuNow = 0
          print("Menu Opsi")
        elif menuNow == 2:
          # menulok2 = menu daya
          menuLok = 2
          menuNow = 0
          print("Menu Daya")
      elif menuLok == 1:
        # masuk menu riwayat
        if menuNow == 0:
          menuLok = 3
          menuNow = 0
        elif menuNow == 1:
          riwayatHps()
      elif menuLok == 2:
        # masuk menu daya
        if menuNow == 0:
          # mematikan daya dipilih iya
          shut_down()

      print(f"menuNow = {menuNow}")
      print(f"menuLok = {menuLok}")

    #mencegah tombol back tertekan terus  
    if GPIO.input(tomBa) and menuLok > 0:
      tekanOkBa = True
      menuLok = 0
      menuNow = 0
      print("Kembali")
      print(f"menuNow = {menuNow}")
      print(f"menuLok = {menuLok}")
  #mencegah tombol ok tertekan terus
  elif not GPIO.input(tomOk):
    tekanOkBa = False
  
    