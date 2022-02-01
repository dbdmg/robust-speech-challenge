import re
from typing import Callable

# Latest version of int2ita => https://gist.github.com/fgiobergia/395fb9b4727790e0a9a2d24d936526f1
def _merge(a: str, b: str, elisione: bool = True, spacing: bool = False, use_e : bool = False) -> str:
    """Utility function to merge two strings, following a bunch of rules,
    as explained in the documentation for int2ita
    """
    if not b:
        return a
    if b[0] in "aeiou" and elisione:
        # avoids "ottantaotto", "ventiuno"
        a = a[:-1]
    if use_e:
        return f"{a} e {b}"
    elif spacing:
        return f'{a} {b}'
    return a + b

def int2ita(n: int, spacing : bool = False, use_e: bool = False, elisione : bool = True) -> str:
    """Convert a number n into its italian string representation (e.g. 42 => quarantadue).
    
    Parameters
    ----------
    n : int
        the number to be converted (currently only numbers < 1 trillion are supported)
    spacing : bool
        whether each "section" of the number should be separated by a space (default False).
        Example (n = 1234):
            * `spacing=True`: mille due cento trenta quattro
            * `spacing=False`: milleduecentotrentaquattro
    use_e : bool
        whether to concatenate separate portions of the numbers with an 'e' (and) (default False)
        Example (n = 1234567):
            * `use_e=True` (`spacing=False`): un milione e duecentotrentaquattromila e cinquecentosessantasette
            * `use_e=False` (`spacing=False`): un milione duecentotrentaquattromilacinquecentosessantasette
    elisione : bool
        whether numbers that require elision (e.g. consecutive vowels) should require elision or not (default True)
        Example (n = 31)
            * `elisione=True` (`spacing=False`): trentuno <= expected behavior in the Italian language
            * `elisione=True` (`spacing=True`): trent uno <= eh # 1
            * `elisione=False` (`spacing=False`): trentauno <= eh # 2
            * `elisione=False` (`spacing=True`): trenta uno <= possibly useful for some language models
    """
    mu = {
        0: "zero",
        1: "uno",
        2: "due",
        3: "tre",
        4: "quattro",
        5: "cinque",
        6: "sei",
        7: "sette",
        8: "otto",
        9: "nove"
    }
    
    m_11_19 = {
        11: "undici",
        12: "dodici",
        13: "tredici",
        14: "quattordici",
        15: "quindici",
        16: "sedici",
        17: "diciassette",
        18: "diciotto",
        19: "diciannove"
    }
    
    m_10_90 = {
        1: "dieci",
        2: "venti",
        3: "trenta",
        4: "quaranta",
        5: "cinquanta",
        6: "sessanta",
        7: "settanta",
        8: "ottanta",
        9: "novanta",
    }

    if n < 10:
        return mu[n]
    elif n < 100:
        d = n // 10
        u = n % 10
        
        if u == 0:
            return m_10_90[d]
        elif d == 1:
            return m_11_19[n]
        else:
            part_a = int2ita(d*10, spacing, use_e, elisione)
            part_b = int2ita(u, spacing, use_e, elisione)
            return _merge(part_a, part_b, elisione, spacing)
    elif n < 1000:
        c = n // 100
        r = n % 100

        part_a = ""
        if c > 1:
            part_a = int2ita(c, spacing, use_e, elisione)
        part_a += f"{' ' if spacing and part_a else ''}cento"
        part_b = int2ita(r, spacing, use_e, elisione) if r > 0 else ""
        return _merge(part_a, part_b, False, spacing)
    elif n < 1_000_000:
        m = n // 1000
        r = n % 1000

        part_a = ""
        if m > 1:
            part_a = int2ita(m, spacing, use_e, elisione) + f"{' ' if spacing else ''}mila"
        else:
            part_a = "mille"
        part_b = int2ita(r, spacing, use_e, elisione) if r > 0 else ""
        return _merge(part_a, part_b, False, spacing, use_e)
    elif n < 1_000_000_000:
        M = n // 1_000_000
        r = n % 1_000_000

        part_a = ""
        if M == 1:
            part_a = "un milione"
        elif M > 1:
            part_a = f"{int2ita(M, spacing, use_e, elisione)} milioni" 
        part_b = int2ita(r, spacing, use_e, elisione) if r > 0 else ""

        return _merge(part_a, part_b, False, True, use_e)
    
    elif n < 1_000_000_000_000:
        M = n // 1_000_000_000
        r = n % 1_000_000_000

        part_a = ""
        if M == 1:
            part_a = "un miliardo"
        elif M > 1:
            part_a = f"{int2ita(M, spacing, use_e, elisione)} miliardi" 
        part_b = int2ita(r, spacing, use_e, elisione) if r > 0 else ""

        return _merge(part_a, part_b, False, True, use_e)
    return "UNSUPPORTED"


def _wrap_i2i(match: re.Match, max_value : int) -> str:
    """ Utility function to map a regex Match (i.e. a number extracted by a regex) to its 'italian' representation.
    """
    num = int(match[1])
    if num > max_value:
        return match[1]
    return int2ita(num, True, False, False)

def replacer(max_value : int = 100_000_000_000) -> Callable[[str], str]:
    """ Returns a function that can be called to convert the numbers within a string.
    Using a closure to compile the regex only once, and then calling .sub() a bunch
    of times -- this speeds up the matching process significantly.
    
    Parameters:
    -----------
        max_value : int
            The largest value that should be encoded as a string. Larger values will
            not be converted.
    
    Returns:
        func : Callable[[str], str]
            A function that gets as input any string of text and returns it with
            numbers converted to their "string" form.
    """
    match_num_re = re.compile(r"(\d+)")
    
    def func(text: str) -> str:
        return match_num_re.sub(lambda x: _wrap_i2i(x, max_value), text)

    return func
