from paramath import generate_expression, tokenize, generate_ast, infix_to_postfix


a = tokenize(
    "(1 - epsilon / (abs(arctan((0 - cos(pi * 5 / 10)) / (sin(pi * 5 / 10) + epsilon)) / pi * 10 + 10 / 2 - 10) + epsilon)) * (arctan((0 - cos(pi * 5 / 10)) / (sin(pi * 5 / 10) + epsilon)) / pi * 10 + 10 / 2) + epsilon / (abs(arctan((0 - cos(pi * 5 / 10)) / (sin(pi * 5 / 10) + epsilon)) / pi * 10 + 10 / 2 - 10) + epsilon) * 0"
)
a = generate_ast(a)
a = infix_to_postfix(a)
print(repr(a).replace("'a'", "a").replace("'b'", "b").replace("'eps'", "eps"))
