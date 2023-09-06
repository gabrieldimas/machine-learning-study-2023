#langkah1 
from PIL import Image
 
img = Image.open('lenna.png')

#langkah2
# Ekstrak setiap channel red, green, blue
r, g, b = img.split()

# Cek panjang ukuran channel red
print(len(r.histogram()))

# Cetak fitur histogram pada channel red
print(r.histogram())