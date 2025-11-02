#!/usr/bin/env python3
# Save as: verify_bst_leaf_nodes.py
# Run tests:      python verify_bst_leaf_nodes.py
# Custom verify:  echo '["insert 2","insert 3","insert 1","delete 2"]' | python verify_bst_leaf_nodes.py
# Or:
#   echo 4 && echo insert 2 && echo insert 3 && echo insert 1 && echo delete 2 | python verify_bst_leaf_nodes.py
# Or:
#   printf "insert 2\ninsert 3\ninsert 1\n" | python verify_bst_leaf_nodes.py

from typing import Optional, List
import sys
import ast

class Node:
    def __init__(self, key: int):
        self.val = key
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None


def getLeafNodes(operations):
    """
    operations: list of strings like "insert 5", "delete 5"
    Returns: space-separated string of leaf values from left to right.
    """

    def insert(root, key):
        if root is None:
            return Node(key)
        cur = root
        while True:
            if key < cur.val:
                if cur.left is None:
                    cur.left = Node(key)
                    break
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = Node(key)
                    break
                cur = cur.right
        return root

    def delete(root, key):
        # Find node and its parent
        parent, cur = None, root
        while cur and cur.val != key:
            parent = cur
            cur = cur.left if key < cur.val else cur.right
        if cur is None:       # key not found; nothing to do (inputs should be valid)
            return root

        def replace_at_parent(parent, old_child, new_child):
            nonlocal root
            if parent is None:
                root = new_child
            elif parent.left is old_child:
                parent.left = new_child
            else:
                parent.right = new_child

        if cur.right is not None:
            # Rule 1: replace node with its right child
            repl = cur.right
            # graft the old left subtree under the leftmost of the replacement
            leftmost = repl
            while leftmost.left is not None:
                leftmost = leftmost.left
            leftmost.left = cur.left
            replace_at_parent(parent, cur, repl)

        elif cur.left is not None:
            # Rule 2: replace node with its left child
            replace_at_parent(parent, cur, cur.left)

        else:
            # Rule 3: delete leaf
            replace_at_parent(parent, cur, None)

        return root

    def collect_leaves_inorder(root):
        ans = []

        def dfs(node):
            if not node:
                return
            dfs(node.left)
            if node.left is None and node.right is None:
                ans.append(str(node.val))
            dfs(node.right)

        dfs(root)
        return " ".join(ans)

    # Build the BST by applying operations
    root = None
    for op in operations:
        op = op.strip()
        if not op:
            continue
        kind, x = op.split()
        x = int(x)
        if kind == "insert":
            root = insert(root, x)
        else:  # "delete"
            root = delete(root, x)

    # Return leaf nodes from left to right
    return collect_leaves_inorder(root)


# ----------------- Helpers to verify ----------------- #

def _parse_ops_from_stdin(text: str) -> List[str]:
    s = text.strip()
    if not s:
        return []
    # Try list literal: ["insert 2","delete 2"]
    if s[0] == '[' and s[-1] == ']':
        return [str(x).strip() for x in ast.literal_eval(s)]
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    # If first line is an int n
    if lines and lines[0].isdigit():
        n = int(lines[0])
        return lines[1:1+n]
    # Otherwise treat each non-empty line as an op
    return lines


def _run_unit_tests():
    def check(ops, expected):
        out = getLeafNodes(ops)
        assert out == expected, f"\nOps: {ops}\nExpected: {expected!r}\nGot: {out!r}"

    # Examples from prompt
    check(["insert 2","insert 3","insert 1"], "1 3")
    check(["insert 2","insert 3","insert 1","delete 2"], "1")

    # Single node insert/delete
    check(["insert 10"], "10")
    check(["insert 10","delete 10"], "")

    # Delete a leaf
    check(["insert 5","insert 3","insert 7","delete 3"], "5 7")

    # Delete a node with only left child
    check(["insert 5","insert 3","insert 2","delete 3"], "2 5")

    # Delete a node with only right child
    check(["insert 5","insert 7","insert 9","delete 7"], "5 9")

    # Delete node with both children (custom rule promotes right child)
    # Tree after inserts:    5
    #                      /   \
    #                     3     8
    #                          / \
    #                         6   9
    # delete 5 -> promote right child (8), graft left subtree (3) to leftmost of promoted right subtree (6)
    # Result leaves (left->right): 3 6 9
    check(["insert 5","insert 3","insert 8","insert 6","insert 9","delete 5"], "3 6 9")

    # Multiple deletions
    check(["insert 4","insert 2","insert 6","insert 1","insert 3","insert 5","insert 7",
           "delete 2","delete 6"], "1 3 5 7")

    # Strictly increasing inserts (degenerate BST), then delete middle
    check(["insert 1","insert 2","insert 3","insert 4","delete 2"], "1 3 4")

    # Strictly decreasing inserts, then delete root repeatedly
    check(["insert 5","insert 4","insert 3","insert 2","insert 1","delete 5","delete 4"], "1 2 3")

    print("All tests passed ✅")


if __name__ == "__main__":
    data = sys.stdin.read()
    if not data.strip():
        _run_unit_tests()
        print("\nTry your own:")
        print('  echo \'["insert 2","insert 3","insert 1","delete 2"]\' | python verify_bst_leaf_nodes.py')
        print('  printf "insert 2\\ninsert 3\\ninsert 1\\n" | python verify_bst_leaf_nodes.py')
    else:
        ops = _parse_ops_from_stdin(data)
        print(getLeafNodes(ops))
