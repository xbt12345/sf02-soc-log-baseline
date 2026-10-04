"""Word tokens plus explicit operator tokens, without labels/source metadata."""
import re

TOKEN=re.compile(r"(?u)\w+|--|>>|&&|\|\||[><|=/\\-]")
OPS={"--":"op_long_option", "-":"op_dash", ">>":"op_append", ">":"op_redirect",
     "<":"op_input", "&&":"op_and", "||":"op_or", "|":"op_pipe",
     "=":"op_equals", "/":"op_slash", "\\":"op_backslash"}


def word_tokens(text):
    return [OPS.get(t,t) for t in TOKEN.findall(text.casefold())]
