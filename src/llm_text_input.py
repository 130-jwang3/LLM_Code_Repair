"""
Pipeline for analyzing code using LLMs with plain code text as input.
"""

def load_code(code_file):
    """
    Loads code from code_file as plain text.
    """
    # TODO: Implement code loading logic
    pass

def analyze_with_llm(code_text):
    """
    Calls the LLM to analyze the code and return bug localization/fix.
    """
    # TODO: Implement LLM call
    pass

if __name__ == "__main__":
    code = load_code("data/processed/sample.py")
    analyze_with_llm(code)