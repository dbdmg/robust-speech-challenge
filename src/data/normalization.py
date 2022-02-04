"""Normalize textual strings"""
from num2words import num2words
import re

#TODO I guess it exists a precompiled list of italian mappings
SYMBOL_TO_STR = {
    "$": "dollari",
    "€": "euro",
    "£": "sterline",
}

def converter(match: re.Match) -> str:
    # adding leading and trailing spaces
    return f" {num2words(int(match[1]), lang='it')} "

def normalize_string(text: str, to_lower : bool = True, expand_symbols : bool = True, convert_numbers : bool = True):
    """Normalize a string"""
    
    if to_lower:
        text = text.lower()
    
    if expand_symbols:
        for sym, t in SYMBOL_TO_STR.items():
            text = text.replace(sym, t)

    if convert_numbers:
        match_num_re = re.compile(r"(\d+)")
        text = match_num_re.sub(converter, text)

    return text
