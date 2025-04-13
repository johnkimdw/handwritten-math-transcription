import torch

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3

DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() else "cpu")

# LaTeX vocabulary mapping
LATEX_VOCAB = {
    # Special tokens
    '<sos>': 0,   # start-of-sequence
    '<eos>': 1,   # end-of-sequence
    '<pad>': 2,   # padding token

    # Digits
    '0': 3,  '1': 4,  '2': 5,  '3': 6,  '4': 7,  
    '5': 8,  '6': 9,  '7': 10, '8': 11, '9': 12,

    # Lowercase letters
    'a': 13, 'b': 14, 'c': 15, 'd': 16, 'e': 17, 'f': 18, 'g': 19, 
    'h': 20, 'i': 21, 'j': 22, 'k': 23, 'l': 24, 'm': 25, 'n': 26, 
    'o': 27, 'p': 28, 'q': 29, 'r': 30, 's': 31, 't': 32, 'u': 33, 
    'v': 34, 'w': 35, 'x': 36, 'y': 37, 'z': 38,

    # Uppercase letters
    'A': 39, 'B': 40, 'C': 41, 'D': 42, 'E': 43, 'F': 44, 'G': 45, 
    'H': 46, 'I': 47, 'J': 48, 'K': 49, 'L': 50, 'M': 51, 'N': 52, 
    'O': 53, 'P': 54, 'Q': 55, 'R': 56, 'S': 57, 'T': 58, 'U': 59, 
    'V': 60, 'W': 61, 'X': 62, 'Y': 63, 'Z': 64,

    # Common mathematical operators and punctuation
    '+': 65, '-': 66, '*': 67, '/': 68, '=': 69, 
    '(': 70, ')': 71, '[': 72, ']': 73, '{': 74, '}': 75, 
    '<': 76, '>': 77, ',': 78, '.': 79, ';': 80, ':': 81,

    # Common LaTeX commands for expressions
    '\\frac': 82,
    '\\sqrt': 83,
    '\\int': 84,
    '\\sum': 85,
    '\\prod': 86,
    '\\lim': 87,
    '\\infty': 88,
    '\\cdot': 89,
    '\\times': 90,
    '\\div': 91,

    # Common trigonometric and logarithmic functions
    '\\sin': 92,
    '\\cos': 93,
    '\\tan': 94,
    '\\log': 95,
    '\\ln': 96,
    '\\exp': 97,
    '\\min': 98,
    '\\max': 99,
    '\\argmin': 100,
    '\\argmax': 101,

    # Delimiters
    '\\left': 102,
    '\\right': 103,

    # Greek letters (lowercase)
    '\\alpha': 104,
    '\\beta': 105,
    '\\gamma': 106,
    '\\delta': 107,
    '\\epsilon': 108,
    '\\zeta': 109,
    '\\eta': 110,
    '\\theta': 111,
    '\\iota': 112,
    '\\kappa': 113,
    '\\lambda': 114,
    '\\mu': 115,
    '\\nu': 116,
    '\\xi': 117,
    '\\omicron': 118,
    '\\pi': 119,
    '\\rho': 120,
    '\\sigma': 121,
    '\\tau': 122,
    '\\upsilon': 123,
    '\\phi': 124,
    '\\chi': 125,
    '\\psi': 126,
    '\\omega': 127,

    # Greek letters (uppercase)
    '\\Gamma': 128,
    '\\Delta': 129,
    '\\Theta': 130,
    '\\Lambda': 131,
    '\\Xi': 132,
    '\\Pi': 133,
    '\\Sigma': 134,
    '\\Upsilon': 135,
    '\\Phi': 136,
    '\\Psi': 137,
    '\\Omega': 138,

    # Additional symbols
    '\\approx': 139,
    '\\neq': 140,
    '\\leq': 141,
    '\\geq': 142,
    '\\subset': 143,
    '\\supset': 144,
    '\\subseteq': 145,
    '\\supseteq': 146,
    '\\cup': 147,
    '\\cap': 148,
    '\\forall': 149,
    '\\exists': 150,
    '\\nabla': 151,
    '\\partial': 152,
    '\\rightarrow': 153,
    '\\leftarrow': 154,
    '\\Rightarrow': 155,
    '\\Leftarrow': 156,
    '\\leftrightarrow': 157,
    '\\Leftrightarrow': 158,
    '\\perp': 159,
    '\\angle': 160,
    '\\degree': 161,

    # Syntactic tokens for environments and structure
    '\\begin': 162,
    '\\end': 163,
    # Additional tokens can be added here if required by the dataset
}

LATEX_VOCAB_SIZE = len(LATEX_VOCAB)
LATEX_PAD_TOKEN = LATEX_VOCAB['<pad>']

# mapping from token --> symbol
LATEX_VOCAB_REVERSE = {idx: token for token, idx in LATEX_VOCAB.items()}
