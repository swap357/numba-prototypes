import ast
import pprint
import symtable
import sys

from dataclasses import dataclass

def attribute_to_qualified_name(node):
    """
    Converts an ast.Attribute node into a fully qualified name string.

    For example, if the AST represents "module.submodule.function",
    this function will return the string "module.submodule.function".

    Args:
        node: An ast.Attribute node or ast.Name node

    Returns:
        str: The fully qualified name as a string
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{attribute_to_qualified_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        return attribute_to_qualified_name(node.func)
    else:
        raise TypeError(f"Expected ast.Attribute or ast.Name, got {type(node).__name__}")


@dataclass
class SymbolInfo:
    name: str
    ast: ast.AST
    calls: list

class NamespaceVisitor(ast.NodeVisitor):

    def __init__(self, source_code, file_name):
        self.source_code = source_code
        self.file_name = file_name
        self.tree = ast.parse(source_code)
        self.symt = symtable.symtable(source_code, file_name , "exec")
        self.namespace_stack = []
        self.functions = {}
        self.global_calls = []

    def update_calls(self, node):
        """Update the calls for a function or register a global call."""
        if self.namespace_stack:
            name = ".".join(self.namespace_stack)
            assert name in self.functions, f"Function {name} not found"
            self.functions[name].calls.append(node)
        else:
            self.global_calls.append(node)

    def visit_all(self):
        """Visit all nodes in the AST."""
        self.visit(self.tree)

    def visit_FunctionDef(self, node):
        """Visit a function definition."""
        # Create a new namespace for the function
        self.namespace_stack.append(node.name)
        name = ".".join(self.namespace_stack)
        self.functions[name] = SymbolInfo(name, node, [])

        # Visit the function body
        self.generic_visit(node)

        # Pop the namespace after visiting the function
        self.namespace_stack.pop()

    def visit_ClassDef(self, node):
        """Visit a class definition."""
        # Create a new namespace for the class
        self.namespace_stack.append(node.name)

        # Visit the class body
        self.generic_visit(node)

        # Pop the namespace after visiting the class
        self.namespace_stack.pop()

    def visit_Call(self, node):
        """Visit a function call."""
        # Update the namespace where this call occurs
        self.update_calls(node.func)

        # Visit the arguments of the function call
        for n in node.args + node.keywords:
            self.generic_visit(n)



def main():
    """Entry point for the compiler driver."""
    if len(sys.argv) < 2:
        print("Usage: python wpc.py <python_source_file>")
        sys.exit(1)

    source_file = sys.argv[1]

    try:
        with open(source_file, 'r') as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"File not found: {source_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Create a NamespaceVisitor instance
    nv = NamespaceVisitor(source_code, source_file)
    nv.visit_all()
    pprint.pp(nv.functions)

    #compiler = CompilerDriver()
    #symbol_table = compiler.compile(source_code)

    #if symbol_table:
    #    print("Symbol Table:")
    #    for symbol, info in symbol_table.items():
    #        print(f"  {symbol}: {info}")
    #
if __name__ == "__main__":
    main()
