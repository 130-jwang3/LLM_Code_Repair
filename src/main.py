import sys
from llm_text_input import analyze_with_llm as text_llm
from llm_graph_input import analyze_with_llm as graph_llm

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py [text|graph] [model_name]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    model = sys.argv[2]

    if mode == "text":
        text_llm(model)
    elif mode == "graph":
        graph_llm(model)
    else:
        print("Invalid mode. Use 'text' or 'graph'.")
