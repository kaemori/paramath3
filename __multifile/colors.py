class Colors:
    class Regular:
        black = "\033[0;30m"
        red = "\033[0;31m"
        green = "\033[0;32m"
        yellow = "\033[0;33m"
        blue = "\033[0;34m"
        purple = "\033[0;35m"
        cyan = "\033[0;36m"
        white = "\033[0;37m"

    class Bold:
        black = "\033[1;30m"
        red = "\033[1;31m"
        green = "\033[1;32m"
        yellow = "\033[1;33m"
        blue = "\033[1;34m"
        purple = "\033[1;35m"
        cyan = "\033[1;36m"
        white = "\033[1;37m"

    class Underline:
        black = "\033[4;30m"
        red = "\033[4;31m"
        green = "\033[4;32m"
        yellow = "\033[4;33m"
        blue = "\033[4;34m"
        purple = "\033[4;35m"
        cyan = "\033[4;36m"
        white = "\033[4;37m"

    class Background:
        black = "\033[40m"
        red = "\033[41m"
        green = "\033[42m"
        yellow = "\033[43m"
        blue = "\033[44m"
        purple = "\033[45m"
        cyan = "\033[46m"
        white = "\033[47m"

    class HighIntensity:
        black = "\033[0;90m"
        red = "\033[0;91m"
        green = "\033[0;92m"
        yellow = "\033[0;93m"
        blue = "\033[0;94m"
        purple = "\033[0;95m"
        cyan = "\033[0;96m"
        white = "\033[0;97m"

    class BoldHighIntensity:
        black = "\033[1;90m"
        red = "\033[1;91m"
        green = "\033[1;92m"
        yellow = "\033[1;93m"
        blue = "\033[1;94m"
        purple = "\033[1;95m"
        cyan = "\033[1;96m"
        white = "\033[1;97m"

    class HighIntensityBackground:
        black = "\033[0;100m"
        red = "\033[0;101m"
        green = "\033[0;102m"
        yellow = "\033[0;103m"
        blue = "\033[0;104m"
        purple = "\033[0;105m"
        cyan = "\033[0;106m"
        white = "\033[0;107m"

    class Reset:
        reset = "\033[0m"


colors = Colors()
