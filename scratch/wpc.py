import ast
import pprint
import symtable
import sys

from collections import defaultdict
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

class CallGraphVisitor(ast.NodeVisitor):

    def __init__(self, source_code, file_name):
        # Stash the arguments
        self.source_code = source_code
        self.file_name = file_name
        # Get the AST once
        self.tree = ast.parse(source_code)
        # Initialize the cpython symtable
        self.symt = symtable.symtable(source_code, file_name , "exec")
        # Filter out all class definitions from the AST
        self.classes = set((node.name for node in ast.walk(self.tree) if
                        isinstance(node, ast.ClassDef)))
        # Setup the namespace and class stacks
        self.namespace_stack = []
        self.class_stack = []
        # Nested dictionary to record class types
        self.class_types = defaultdict(dict)
        # Dictionary to record all functions
        self.functions = {}
        # List of all global ast.Call nodes
        self.global_calls = []

    def get_call_graph(self) -> dict[str: tuple[str]]:
        """Obtain a call graph suitable for processing with networkx.

        Returns a dictionary mapping function names as strings to lists of
        function names as strings.
        """

        return {k:tuple(c[1] for c in v.calls) for k,v in self.functions.items()}

    def update_calls(self, node):
        """Update the calls for a function or register a global call."""
        # Flatten the name of the call from ast.Attribute or ast.Name
        call_qname = attribute_to_qualified_name(node)
        class_name = self.class_stack[-1]
        if call_qname.startswith("self"):
            # If the call starts with "self", it is a method call, we replace
            # the "self" with the current class name to qualify it.
            call_qname = class_name + call_qname[4:]
        if class_name and call_qname.startswith(class_name):
            # Replace calls from class attributes with their qualified name.
            # First split the qualified name by the dot separator.
            split_qname = call_qname.split(".")
            # Get the types of the currentclasses attributes.
            current_class_types = self.class_types[class_name]
            # If the second element in the qualified name matches the name of
            # the class attribute, replace the reference to the class.attribute
            # string with the correct type.
            if split_qname[1] in current_class_types:
                call_qname = ".".join([current_class_types[split_qname[1]]] +
                                      split_qname[2:])
        if call_qname in self.classes:
            # If the call ends with the current class name, we replace it with
            # the constructor call, since this is the Python semantics.
            call_qname = call_qname + ".__init__"

        if self.namespace_stack:
            name = ".".join(self.namespace_stack)
            assert name in self.functions, f"Function {name} not found"
            self.functions[name].calls.append((node, call_qname))
        else:
            self.global_calls.append((node, call_qname))

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
        # Push the name of the class onto the class_stack
        self.class_stack.append(node.name)

        # Visit the class body
        self.generic_visit(node)

        # Pop the namespace after visiting the class
        self.namespace_stack.pop()
        # Pop the class name from the class_stack
        self.class_stack.pop()

    def visit_Call(self, node):
        """Visit a function call."""
        # Update the namespace where this call occurs
        self.update_calls(node.func)

        # Visit the arguments of the function call
        for n in node.args + node.keywords:
            self.generic_visit(n)

    def visit_AnnAssign(self, node):
        """Visit an annotated assignment."""
        if self.class_stack[-1] == self.namespace_stack[-1]:
            # Class and namespace stack have the identical last value. This
            # means we are in a class definition.
            class_name = self.class_stack[-1]
            assert isinstance(node.target, ast.Name)
            attribute_name = node.target.id
            assert isinstance(node.annotation, ast.Name)
            attribute_type = node.annotation.id
            # Populate the class_type datastructure
            self.class_types[class_name][attribute_name] = attribute_type



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
    cgv = CallGraphVisitor(source_code, source_file)
    cgv .visit_all()
    print("########## Symbol Table ##########")
    pprint.pp(cgv.functions)
    print("########## ------------ ##########")
    print("########## Global Calls ##########")
    pprint.pp(cgv.global_calls)
    print("########## ------------ ##########")
    return cgv

    #compiler = CompilerDriver()
    #symbol_table = compiler.compile(source_code)

    #if symbol_table:
    #    print("Symbol Table:")
    #    for symbol, info in symbol_table.items():
    #        print(f"  {symbol}: {info}")
    #
if __name__ == "__main__":
    cgv = main()
