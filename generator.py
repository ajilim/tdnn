import random
import math
with open("g_data.txt",'w') as f:
    for i in range(1,100000):
        x=random.randint(1,100)
        y=round(math.sin(x))
        f.write(str(x))
        f.write(" ")
        if y==0:
            f.write("0 1 0")
        elif y==1:
            f.write("0 0 1")
        else:
            f.write("1 0 0")
        f.write('\n')
with open("g_data_test.txt",'w') as f:
    for i in range(1,500):
        x = random.randint(1, 100)
        y = round(math.sin(x))
        f.write(str(x))
        f.write(" ")
        if y==0:
            f.write("0 1 0")
        elif y==1:
            f.write("0 0 1")
        else:
            f.write("1 0 0")
        f.write('\n')