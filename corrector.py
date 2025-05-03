import os
import openai
from dotenv import load_dotenv

# few‐shot prompt template
BASE_PROMPT = """
You are a LaTeX correction assistant.
Your job is: Given a possibly broken LaTeX string, return the *only* corrected LaTeX.
Do NOT add any commentary or extra words.
Be very strict about matching braces, subscripts, superscripts, and fraction order.

### Examples

Input:  \\int^{{1}} x^2 dx  
Output: \\int_{{0}}^{{1}} x^2 \\, dx

Input:  \\frac{{\\frac{{2}}{{3}}}}{{1}}  
Output: \\frac{{1}}{{\\frac{{2}}{{3}}}}

Input:  a_{{n}}=k l_{{n}}\\cdot\\frac{{b_{{n}}}}{{b_{{a}}}}\\cdot\\frac{{s_{{n}}}}{{s_{{a}}}}  
Output: a_{{n}} = k\\,l_{{n}}\\cdot\\frac{{b_{{n}}}}{{b_{{a}}}}\\cdot\\frac{{s_{{n}}}}{{s_{{a}}}}

Input: \\Lambda{id}=\\chi(X)
Output: \\Lambda\\text{{id}}=\\chi(X)

### Important rule.
RETURN ONLY THE LATEX. DO NOT RETURN ANYTHING ELSE.
### Now correct this:

Input:  {raw}
LaTeX:
""".strip()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "key not in .env file")
client = openai.OpenAI(api_key=api_key)

def _correct_latex_raw(raw: str,
                  model: str = "gpt-4.1-nano",
                  temperature: float = 0.0,
                  max_tokens: int = 256) -> str:
    """
    Corrects potentially malformed LaTeX output from the Seq2Seq model.
    
    Args:
        raw: Raw LaTeX output from Seq2Seq model
        model: OpenAI model to use
        temperature: Control randomness (0.0 = deterministic)
        max_tokens: Maximum output length
        
    Returns:
        Corrected LaTeX string
    """
    if not raw or raw.strip() == "": return raw
        
    prompt = BASE_PROMPT.replace("{raw}", raw)
    
    response = client.chat.completions.create(
        model=model, 
        messages=[
            {"role": "system", "content": "You are a LaTeX correction assistant. Return ONLY the corrected LaTeX without any explanation."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # extract and clean response
    corrected = response.choices[0].message.content.strip()
    
    if corrected.startswith("LaTeX:"): corrected = corrected[6:].strip()
    
    return corrected


_correction_cache = {}

# cached to reduce api calls during testing
def correct_latex(raw: str, **kwargs) -> str:
    if raw in _correction_cache:
        return _correction_cache[raw]
    fixed = _correct_latex_raw(raw, **kwargs)
    _correction_cache[raw] = fixed
    return fixed