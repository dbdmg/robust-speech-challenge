"""Normalize textual strings"""

#TODO I guess it exists a precompiled list of italian mappings
SYMBOL_TO_STR = {
    "$": "dollari",
    "€": "euro",
    "£": "sterline",
}

def normalize_string(text: str, to_lower=True, expand_symbols=True):
    """Normalize a string"""
    
    if to_lower:
        new = text.lower()
    
    if expand_symbols:
        for sym, t in SYMBOL_TO_STR.items():
            new = new.replace(sym, t)

    return new