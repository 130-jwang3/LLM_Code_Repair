from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os


# returns list of dictionaries, first one being file tree, the rest will be file sections and its contents
def split_text(input, chunk_size):
    separated_text = []
    # input will be a json file
    try:
        with open(input, "r", encoding="utf-8") as f:
            src = json.load(f)

        # explicit telemetry helps
        print(f"[split_text] loading: {os.path.abspath(input)}  chunk_size={chunk_size}")
        n_files = len(src.get("files", []))
        print(f"[split_text] bundle files[]: {n_files}")

        # build splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=50,
        )

        # keep your original first entry
        separated_text.append({"file_tree": src.get("file_tree", "")})

        # produce chunks; add line spans so caller can map to file-global lines
        for i in src.get("files", []):
            path = i.get("path")
            content = i.get("content", "") or ""
            if not path or not content:
                continue

            chunks = text_splitter.split_text(content)
            if not isinstance(chunks, list):
                print(f"[split_text] WARN: splitter returned non-list for {path}")
                continue

            scan_pos = 0  # <-- NEW: move forward through the file as we match chunks
            for j, sec in enumerate(chunks, start=1):
                try:
                    # prefer matching at/after last end to avoid earlier duplicate hits
                    start_off = content.find(sec, scan_pos)
                    if start_off == -1:
                        # fallback: naive locate
                        start_off = content.find(sec)
                    end_off = start_off + len(sec) if start_off != -1 else None

                    if start_off is None or start_off == -1 or end_off is None:
                        # fallback to whole-file span if mapping fails
                        start_line = 1
                        end_line = content.count("\n") + 1
                    else:
                        # precise line calc by counting newlines before offsets
                        start_line = content.count("\n", 0, start_off) + 1
                        end_line = content.count("\n", 0, end_off) + 1
                        # advance cursor for next search
                        scan_pos = end_off
                except Exception as e:
                    print(f"[split_text] WARN: line map failed for {path} sec#{j}: {e}")
                    start_line = 1
                    end_line = content.count("\n") + 1

                separated_text.append({
                    "file": path,
                    "section": j,
                    "content": sec,
                    "start_line": start_line,
                    "end_line": end_line,
                    "num_lines": sec.count("\n") + 1
                })

    except FileNotFoundError:
        print(f"Error: {input} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON")
    except Exception as e:
        # catch-all so we don't silently return just the file_tree
        print(f"[split_text] ERROR: {e}")

    # telemetry
    file_chunks = sum(1 for x in separated_text if isinstance(x, dict) and "file" in x)
    print(f"[split_text] produced entries: total={len(separated_text)} file_chunks={file_chunks}")

    return separated_text


def split_ast(input, chunk_size):
    pass

# example usage of split_text
# def main():

#     separated = split_text("LLM_Code_Repair/data/text/repo_text_bundle.json", 4000)
#     print(len(separated))
#     print(separated[99])

# main()
