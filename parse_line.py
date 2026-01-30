file = """
modulo_i = better_mod(i)
return if(modulo_i != 0, modulo_i, -10)
return modulo_i + 2""".strip()


def tokenize(file):
    splitters = ["(", ")", ",", "+", "-", "*", "/", "%", "=", "\n"]
    tokens = []
    for splitter in splitters:
        file = file.replace(splitter, f" {splitter} ")
    while "  " in file:
        file = file.replace("  ", " ")
    return file.split(" ")


print(tokenize(file))

for line in file.split("\n"):
    pass
