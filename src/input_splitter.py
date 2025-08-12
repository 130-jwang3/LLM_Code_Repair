from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

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
    
    pass

# example usage of split_text
# def main():

#     separated = split_text("LLM_Code_Repair/data/text/repo_text_bundle.json", 4000)
#     print(len(separated))
#     print(separated[99])

# main()