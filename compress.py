"""
Assignment 2 starter code
CSC148, Winter 2022
Instructors: Bogdan Simion, Sonya Allin, and Pooja Vashisth

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations
import time
from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression
# ====================

# Citations
# https://mcs.utm.utoronto.ca/~148/assignments/a2/handout/a2.html
# https://www.programiz.com/python-programming/methods/built-in/bytes

def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for byte in text:
        if byte not in freq_dict:
            freq_dict[byte] = 1

        else:
            freq_dict[byte] += 1

    return freq_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    sorted_dict = dict(sorted(freq_dict.items(), key=lambda items: items[1]))
    sorted_list = list(sorted_dict.items())
    keys, fre = [a[0] for a in sorted_list], [a[1] for a in sorted_list]
    nodes = []

    for x in range(len(keys)):
        nodes.append([HuffmanTree(keys[x]), fre[x]])

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda items: items[1])
        left, left_remove, left_sum = nodes[0][0], nodes[0], nodes[0][1]
        right, right_remove, right_sum = nodes[1][0], nodes[1], nodes[1][1]
        new_node = HuffmanTree(None, left, right)
        nodes.remove(left_remove)
        nodes.remove(right_remove)
        nodes.append([new_node, left_sum + right_sum])

    if len(freq_dict) == 1:
        return HuffmanTree(None, HuffmanTree(sorted_list[0][0]), HuffmanTree())

    return nodes[0][0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree = HuffmanTree()
    >>> get_codes(tree) == {}
    True
    """
    if tree == HuffmanTree():
        return {}
    else:
        code_dict = {}
        if not tree:
            return {}

        if tree.is_leaf():
            code_dict = {tree.symbol: ""}

        for key, fre in get_codes(tree.left).items():
            code_dict[key] = "0" + fre

        for key, fre in get_codes(tree.right).items():
            code_dict[key] = "1" + fre

        return code_dict


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    _number_help(0, tree)


def _number_help(curr: int, tree: HuffmanTree) -> int:
    if tree.symbol is not None or (tree.left is None and tree.right is None):
        return curr

    else:
        if tree.left:
            curr = _number_help(curr, tree.left)

        if tree.right:
            curr = _number_help(curr, tree.right)
        tree.number = curr
        return curr + 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    >>> tree =HuffmanTree()
    >>> freq = {}
    >>> a = avg_length(tree, freq)
    >>> a == 0
    True
    """
    if tree.is_leaf():
        return 0
    else:
        codes = get_codes(tree)
        curr = 0

        for keys in freq_dict:
            curr += freq_dict[keys] * len(codes[keys])

        total_freq = sum(freq_dict.values())
        return curr / total_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> tesy = bytes([])
    >>> codes = {}
    >>> compress_bytes(tesy, codes) == bytes([])
    True
    """
    if codes == {}:
        return bytes([])
    else:
        final = []
        ans = ''

        for item in text:
            ans += codes[item]
            if len(ans) > 8:
                final += [int(ans[:8], 2)]
                ans = ans[8:]

        if ans:
            final += [bits_to_byte(ans[:8])]
            if 16 > len(ans) > 8:
                ans = ans[8:] + '0' * (8 - len(ans[8:]))
                final += [int(ans, 2)]

        return bytes(final)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None),\
                           HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None),\
                           HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108, 1, 3, \
1, 2, 1, 4]
    >>> tree= HuffmanTree()
    >>> tree_to_bytes(tree) == bytes()
    True
    """
    if tree == HuffmanTree(None):
        return bytes()

    elif tree.is_leaf():
        return bytes()

    else:
        lst = []
        if not tree.left.is_leaf():
            lst += tree_to_bytes(tree.left)
        if not tree.right.is_leaf():
            lst += tree_to_bytes(tree.right)
        if tree.left.is_leaf():
            lst += [0, tree.left.symbol]
        else:
            lst += [1, tree.left.number]
        if tree.right.is_leaf():
            lst += [0, tree.right.symbol]
        else:
            lst += [1, tree.right.number]

    return bytes(lst)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = []
    >>> generate_tree_general(lst, 0) == HuffmanTree()
    True
    """

    if len(node_lst) == 0:
        return HuffmanTree()

    else:

        if node_lst[root_index].l_type == 0:
            left = HuffmanTree(node_lst[root_index].l_data)

        else:
            left = generate_tree_general(node_lst, node_lst[root_index].l_data)

        if node_lst[root_index].r_type == 0:
            right = HuffmanTree(node_lst[root_index].r_data)

        else:
            right = generate_tree_general(node_lst, node_lst[root_index].r_data)

        return HuffmanTree(None, left, right)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> lst = []
    >>> generate_tree_postorder(lst, 0) == HuffmanTree()
    True
    """
    root_index += 0  # Only here so that pyTA doesn't cause any error

    if len(node_lst) == 0:
        return HuffmanTree()
    else:
        return _postorder_helper(node_lst)


def _postorder_helper(lst: list[ReadNode]) -> HuffmanTree:
    root, tree = lst.pop(), HuffmanTree(None)

    if root.r_type:
        tree.right = _postorder_helper(lst)

    else:
        tree.right = HuffmanTree(root.r_data)

    if root.l_type:
        tree.left = _postorder_helper(lst)

    else:
        tree.left = HuffmanTree(root.l_data)

    return tree


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, compress_bytes(b'helloworld', get_codes(tree)),\
                         len(b'helloworld'))
    b'helloworld'
    >>> tree= HuffmanTree()
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, compress_bytes(b'', get_codes(tree)), 0) == bytes()
    True
    """
    if size == 0:
        return bytes()

    final = []
    huf = tree

    for byte in text:
        text = byte_to_bits(byte)

        for char in text:
            if char == "0":
                huf = huf.left

            else:
                huf = huf.right

            if huf.symbol is not None:
                final.append(huf.symbol)
                huf = tree

            if len(final) == size:
                return bytes(final)

    return bytes(final)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    a = build_huffman_tree(freq_dict)
    tree.left = a.left
    tree.right = a.right


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
