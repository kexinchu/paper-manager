import re

class RadixNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.label = None  # Optional: e.g., "EMAIL", "PHONE"

class RadixTree:
    def __init__(self):
        self.root = RadixNode()
    
    def insert(self, pattern: str, label: str):
        node = self.root
        for c in pattern:
            if c not in node.children:
                node.children[c] = RadixNode()
            node = node.children[c]
        node.is_end = True
        node.label = label

    def match_prefix(self, s: str, start: int) -> tuple[int, str]:
        """Return end position and label if matched."""
        node = self.root
        i = start
        last_match = -1
        last_label = None
        while i < len(s) and s[i] in node.children:
            node = node.children[s[i]]
            if node.is_end:
                last_match = i
                last_label = node.label
            i += 1
        if last_match != -1:
            return last_match + 1, last_label  # end idx is exclusive
        return None


def detect_blocks(prompt: str, radix_tree: RadixTree):
    blocks = []
    i = 0
    while i < len(prompt):
        match = radix_tree.match_prefix(prompt, i)
        if match:
            end_idx, label = match
            blocks.append({
                "text": prompt[i:end_idx],
                "start": i,
                "end": end_idx,
                "privacy": True,
                "label": label
            })
            i = end_idx
        else:
            # 扫描非敏感块
            j = i + 1
            while j < len(prompt):
                if radix_tree.match_prefix(prompt, j):
                    break
                j += 1
            blocks.append({
                "text": prompt[i:j],
                "start": i,
                "end": j,
                "privacy": False
            })
            i = j
    return blocks


if __name__ == "__main__":
    prompt = "Hello, my email is john.doe@gmail.com and my number is 123-456-7890."
    radix_tree = RadixTree()
    radix_tree.insert("john.doe@gmail.com", "EMAIL")
    radix_tree.insert("123-456-7890", "PHONE")

    blocks = detect_blocks(prompt, radix_tree)

    for blk in blocks:
        print(blk)
