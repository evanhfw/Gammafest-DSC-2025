import os


def read_paper_contents(path):
    file_list = os.listdir(path)
    txt_files = sorted([f for f in file_list if f.endswith(".txt")])

    contents = []

    for file_name in txt_files:
        file_path = os.path.join(path, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                contents.append(content)
                print(f"Successfully read file {file_name}")
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            continue

    print(f"Total papers read: {len(contents)}")

    return contents
