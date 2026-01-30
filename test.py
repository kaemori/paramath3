from math import modf

for i in range(10):
    print(i % 3)

    # mod 3 but not using &
    print(modf(i / 3)[0] * 3)
