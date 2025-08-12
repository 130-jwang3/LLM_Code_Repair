from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
import tiktoken
import json

# returns list of dictionaries, first one being file tree, the rest will be file sections and its contents
def split_text(input, chunk_size):
    separated_text = []
    #input will be a json file 
    try:
        with open(input) as f:
            src = json.load(f)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder (
            encoding_name='cl100k_base',
            chunk_size=chunk_size,
            chunk_overlap=50,
        )

        separated_text.append({'file_tree': src['file_tree']})

        for i in src['files']:
            chunks = text_splitter.split_text(i['content'])
         
            for j, sec in enumerate(chunks):
                separated_text.append({'file' : i['path'], 'section' : j+1, 'content' : sec})


    except FileNotFoundError:
        print(f"Error: {input} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON ")

    finally:
        return separated_text

def split_ast(input, chunk_size):
    modules = {}
    results = []
    try: 
        with open(input) as f:
            src = json.load(f)['nodes']

        #first group up the nodes by module
        for node in src:
            module = node['module']
            if module not in modules:
                modules[module] = []
            modules[module].append(node)

        #will treat the group of nodes as text and split by the border of the nodes 
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder (
            encoding_name='cl100k_base',
            chunk_size=chunk_size,
            chunk_overlap = 50,
            separators= ["}, {"]
        )

        #if module is too big, it will be split with labeled sections 
        for k,v in modules.items():
            chunks = splitter.split_text(json.dumps(v))
            for j, sec in enumerate(chunks):
                results.append({"module": k, "section" : j+1, "nodes": sec})
        print(len(results))             

    except FileNotFoundError:
        print(f"Error: {input} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON ")
    

    return results

# example usage of split_text
# def main():

#     nodes = split_ast("data/graphs/graph.json", 4000)
#     print(nodes[49])
#     print('\n')
#     print(nodes[50])
#     print('\n')
#     print(nodes[51])


#     separated = split_text("LLM_Code_Repair/data/text/repo_text_bundle.json", 4000)
#     print(len(separated))
#     print(separated[99])


# main()