from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(input):
    with open(input) as f:
        src = f.read()
    text_splitter = RecursiveCharacterTextSplitter (
        chunk_size=4000,
        chunk_overlap=10
    )
    separated_text = text_splitter.create_documents([src])
    return separated_text


# example usage of split_text
# def main():

#     separated = split_text("./data/repo_summary/pygithub-src.txt")
#     print(len(separated))
#     print(separated[439].page_content)
# main()