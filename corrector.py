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

### Important rule.
RETURN ONLY THE LATEX. DO NOT RETURN ANYTHING ELSE.

### Now correct this:

Input:  {raw}
LaTeX:
""".strip()

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY", "key not in .env file")
client = openai.OpenAI(api_key=api_key)


def correct_latex(raw: str,
                  model: str = "gpt-4.1-nano",
                  temperature: float = 0.0,
                  max_tokens: int = 256) -> str:
    """
    raw: LaTeX output from our model
    model: gpt chat model
    temperature: deterministic
    max_tokens: limit for corrected output
    
    returns: hopefully corrected string
    """
    prompt = BASE_PROMPT.format(raw=raw)
    
    # Use the carefully crafted prompt instead of a simple instruction
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
    
    # additional cleanup to remove potential "LaTeX:" prefix
    # if corrected.startswith("LaTeX:"):
    #     corrected = corrected[6:].strip()
    
    return corrected

if __name__ == "__main__":
    # Example usage:
    raw_outputs = [
        r"\int^{1} x^2 dx",         # \int_{0}^{1} x^2 \, dx
        r"\frac{\frac{2}{3}}{1}",   # \frac{1}{\frac{2}{3}}
        r"\Lambda{id}=\chi(X)"      # \Lambda\{id\}=\chi(X)
    ]
    for raw in raw_outputs:
        fixed = correct_latex(raw)
        print(f"Raw:    {raw}\nFixed:  {fixed}\n")