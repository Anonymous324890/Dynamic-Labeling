def readd(ff):
    with open(ff) as f:
        lines = f.readlines()

    return float(lines[-2].strip().split()[-1])

d = []
for i in range(10):
    d.append(readd(f"{i}.txt"))

print (str(d))
