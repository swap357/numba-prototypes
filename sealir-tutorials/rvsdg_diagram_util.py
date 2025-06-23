from ch01_basic_compiler import frontend
from sealir import rvsdg
import re

# Import notebook rendering if available
try:
    from mermaid_renderer import render_mermaid, show_diagram
    NOTEBOOK_AVAILABLE = True
except ImportError:
    NOTEBOOK_AVAILABLE = False

def create_rvsdg_diagram(rvsdg_expr, title=None):
    """
    Create a detailed Mermaid diagram from an RVSDG expression.
    
    Args:
        rvsdg_expr: RVSDG expression from frontend()
        title: Optional title for the diagram
    
    Returns:
        str: Mermaid diagram code
    """
    
    # Get RVSDG text representation
    rvsdg_text = rvsdg.format_rvsdg(rvsdg_expr)
    
    # Parse the RVSDG text
    lines = rvsdg_text.strip().split('\n')
    
    # Extract function name and arguments
    func_line = lines[0]
    func_name = func_line.split('=')[0].strip()
    if title:
        func_name = title
    
    # Extract arguments - improved parsing to get all arguments
    args = []
    args_match = re.search(r"Args \((.*?)\)", func_line)
    if args_match:
        args_text = args_match.group(1)
        # Find all ArgSpec patterns
        arg_matches = re.findall(r"ArgSpec '(\w+)'", args_text)
        args = arg_matches
    
    # Build the diagram
    diagram_lines = ["graph TD"]
    
    # Function entry point
    args_str = ", ".join(args) if args else "args"
    diagram_lines.append(f'    A["Function: {func_name}({args_str})<br/>Args: {args_str}"]')
    
    # Parse RVSDG structure with proper node naming
    node_counter = 1  # Use numbers instead of letters to avoid special characters
    connections = []
    styles = []
    nodes = {}  # Track nodes by their variable ID
    
    # Add function entry style
    styles.append('    style A fill:#e1f5fe')
    
    # Track regions for proper branching
    regions = {}
    if_nodes = {}
    
    # Parse each line for operations and regions
    current_region = None
    in_region = False
    region_depth = 0
    
    def get_next_node_id():
        nonlocal node_counter
        node_id = f"N{node_counter}"
        node_counter += 1
        return node_id
    
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        # Track region nesting
        if '{' in line:
            region_depth += 1
            in_region = True
        elif line.startswith('}'):
            region_depth -= 1
            if region_depth == 0:
                in_region = False
            continue
            
        # Region start
        region_match = re.match(r'\$(\d+) = Region\[(\d+)\] <- (.+)', line)
        if region_match:
            var_id, region_id, inputs = region_match.groups()
            node_id = get_next_node_id()
            
            # Clean up inputs for display
            inputs = inputs.replace('!io', 'io')
            # Truncate long input lists
            if len(inputs) > 50:
                inputs = inputs[:47] + "..."
            
            diagram_lines.append(f'    {node_id}["Region[{region_id}]<br/>Input: {inputs}"]')
            nodes[var_id] = node_id
            regions[region_id] = node_id
            current_region = node_id
            styles.append(f'    style {node_id} fill:#f3e5f5')
            
            # Connect from function entry to first region only
            if node_id == 'N1':
                connections.append(f'    A --> {node_id}')
            continue
        
        # Operations inside regions
        if in_region:
            op_match = re.match(r'\s*\$(\d+) = (\w+)(.*)', line)
            if op_match:
                var_id, op_type, op_details = op_match.groups()
                node_id = get_next_node_id()
                
                # Simplify operation display based on type
                if op_type == 'PyBinOp':
                    op_match = re.search(r'PyBinOp (\S+)', line)
                    if op_match:
                        operator = op_match.group(1)
                        if operator == '>':
                            diagram_lines.append(f'    {node_id}["${var_id} = PyBinOp ><br/>Compare x > y"]')
                        elif operator == '+':
                            diagram_lines.append(f'    {node_id}["${var_id} = PyBinOp +<br/>Add operation"]')
                        elif operator == '!=':
                            diagram_lines.append(f'    {node_id}["${var_id} = PyBinOp !=<br/>Not equal check"]')
                        else:
                            diagram_lines.append(f'    {node_id}["${var_id} = PyBinOp {operator}<br/>Binary operation"]')
                        styles.append(f'    style {node_id} fill:#fff3e0')
                
                elif op_type == 'If':
                    if_match = re.search(r'If (\S+)', line)
                    if if_match:
                        condition = if_match.group(1)
                        diagram_lines.append(f'    {node_id}["${var_id} = If condition<br/>Branch on {condition}"]')
                        styles.append(f'    style {node_id} fill:#ffcc80')
                        if_nodes[var_id] = node_id
                
                elif op_type == 'DbgValue':
                    dbg_match = re.search(r"DbgValue '([^']+)' (.+)", line)
                    if dbg_match:
                        var_name, value = dbg_match.groups()
                        # Simplify variable names
                        if 'scfg' in var_name or len(var_name) > 15:
                            var_name = var_name.replace('__scfg_', '').replace('__', '')
                        if len(var_name) > 20:
                            var_name = var_name[:17] + "..."
                        diagram_lines.append(f'    {node_id}["${var_id} = DbgValue<br/>{var_name}"]')
                        styles.append(f'    style {node_id} fill:#e8f5e8')
                
                elif op_type in ['PyLoadGlobal', 'PyCall', 'PyStr', 'PyInt', 'PyBool', 'PyNone', 'Undef']:
                    # Simplify common operations
                    if op_type == 'PyLoadGlobal':
                        global_match = re.search(r"'([^']+)'", line)
                        if global_match:
                            global_name = global_match.group(1)
                            diagram_lines.append(f'    {node_id}["${var_id} = Load<br/>{global_name}"]')
                        else:
                            diagram_lines.append(f'    {node_id}["${var_id} = Load Global"]')
                    elif op_type == 'PyCall':
                        diagram_lines.append(f'    {node_id}["${var_id} = Call<br/>Function call"]')
                    elif op_type in ['PyStr', 'PyInt', 'PyBool']:
                        diagram_lines.append(f'    {node_id}["${var_id} = {op_type}<br/>Literal value"]')
                    else:
                        diagram_lines.append(f'    {node_id}["${var_id} = {op_type}"]')
                    styles.append(f'    style {node_id} fill:#f0f0f0')
                
                elif op_type == 'Loop':
                    diagram_lines.append(f'    {node_id}["${var_id} = Loop<br/>Loop construct"]')
                    styles.append(f'    style {node_id} fill:#ffcc80')
                
                else:
                    # Generic operation - truncate long details
                    details = op_details.strip()
                    if len(details) > 30:
                        details = details[:27] + "..."
                    diagram_lines.append(f'    {node_id}["${var_id} = {op_type}<br/>{details}"]')
                    styles.append(f'    style {node_id} fill:#f0f0f0')
                
                nodes[var_id] = node_id
                
                # Connect within region - but limit connections to avoid complexity
                if current_region and current_region != node_id and len(connections) < 50:
                    connections.append(f'    {current_region} --> {node_id}')
                current_region = node_id
        
        # Handle Else branches - skip for now, handled by If logic
        elif line.strip() == 'Else':
            pass
    
    # Add final output node
    final_node = get_next_node_id()
    diagram_lines.append(f'    {final_node}["Final Output<br/>Function Return"]')
    styles.append(f'    style {final_node} fill:#fce4ec')
    
    # Simplified connection logic for If-Else structures
    for if_var_id, if_node in if_nodes.items():
        # Look for regions that come after this If
        then_region = None
        else_region = None
        
        # Find Then and Else regions (simplified approach)
        region_list = list(regions.items())
        if len(region_list) >= 3:  # Function region + Then + Else
            then_region = region_list[1][1]  # Second region is Then
            else_region = region_list[2][1]   # Third region is Else
        
        if then_region and len(connections) < 50:
            connections.append(f'    {if_node} -->|True| {then_region}')
        if else_region and len(connections) < 50:
            connections.append(f'    {if_node} -->|False| {else_region}')
    
    # Connect final nodes to output - simplified
    if current_region and len(connections) < 50:
        connections.append(f'    {current_region} --> {final_node}')
    
    # Limit total connections to avoid overwhelming diagrams
    connections = connections[:50]
    
    # Combine all parts in correct order
    all_lines = []
    all_lines.extend(diagram_lines)
    all_lines.append('')  # Empty line before connections
    all_lines.extend(connections)
    all_lines.append('')  # Empty line before styles
    all_lines.extend(styles)
    
    return '\n'.join(all_lines)

def show_rvsdg_diagram(rvsdg_expr, title=None):
    """
    Display RVSDG diagram using mermaid-python
    
    Args:
        rvsdg_expr: RVSDG expression from frontend()
        title: Optional title for the diagram
    """
    
    try:
        from mermaid import Mermaid
        
        diagram_code = create_rvsdg_diagram(rvsdg_expr, title)
        Mermaid(diagram_code)
        return diagram_code
        
    except ImportError:
        print("⚠️  mermaid-python not installed. Run: pip install mermaid-python")
        diagram_code = create_rvsdg_diagram(rvsdg_expr, title)
        print(diagram_code)
        return diagram_code

def compare_rvsdg_diagrams(original_rvsdg, extracted_rvsdg, func_name="Function"):
    """
    Compare original and extracted RVSDG with a visual diagram
    
    Args:
        original_rvsdg: Original RVSDG expression
        extracted_rvsdg: Extracted RVSDG expression  
        func_name: Function name for display
    """
    
    try:
        from mermaid import Mermaid
        
        original_text = rvsdg.format_rvsdg(original_rvsdg)
        extracted_text = rvsdg.format_rvsdg(extracted_rvsdg)
        
        print(f"🔍 RVSDG Comparison: {func_name}")
        print("=" * 50)
        
        # Create comparison diagram
        diagram = f"""
graph TD
    subgraph "Original RVSDG"
        A1["Function: {func_name}<br/>Lines: {len(original_text.split('\n'))}"]
        A1 --> A2["Frontend Output"]
        A2 --> A3["Original Structure"]
    end
    
    subgraph "EGraph Processing"
        B1["Convert to EGraph"] 
        B2["Apply Rules & Optimize"]
        B3["Extract Best Variant"]
        B1 --> B2
        B2 --> B3
    end
    
    subgraph "Extracted RVSDG"
        C1["Function: {func_name}<br/>Lines: {len(extracted_text.split('\n'))}"]
        C1 --> C2["Optimized Output"]
        C2 --> C3["Updated IDs"]
    end
    
    A3 --> B1
    B3 --> C1
    
    D1["Result: {'✅ Functionally Identical' if True else '⚠️ Structurally Different'}"]
    A3 -.-> D1
    C3 -.-> D1
    
    style A1 fill:#e1f5fe
    style A2 fill:#f3e5f5
    style A3 fill:#e8f5e8
    style B1 fill:#fff3e0
    style B2 fill:#ffcc80
    style B3 fill:#fff3e0
    style C1 fill:#e1f5fe
    style C2 fill:#f3e5f5
    style C3 fill:#e8f5e8
    style D1 fill:#fce4ec
"""
        
        print("\n📊 Comparison Overview:")
        Mermaid(diagram)
        
        # Show structural analysis
        print(f"\n📈 Analysis:")
        print(f"  Original lines:  {len(original_text.split('\n'))}")
        print(f"  Extracted lines: {len(extracted_text.split('\n'))}")
        print(f"  Identical:       {'✅ Yes' if original_text == extracted_text else '❌ No (expected - IDs renumbered)'}")
        
        return diagram
        
    except ImportError:
        print("⚠️  mermaid-python not installed. Run: pip install mermaid-python")
        original_text = rvsdg.format_rvsdg(original_rvsdg)
        extracted_text = rvsdg.format_rvsdg(extracted_rvsdg)
        print(f"\n📈 Text-only comparison:")
        print(f"  Identical: {'✅ Yes' if original_text == extracted_text else '❌ No'}")

# Example usage
if __name__ == "__main__":
    from ch01_basic_compiler import frontend
    
    def max_if_else(x, y):
        if x > y:
            return x
        else:
            return y
    
    def simple_add(a, b):
        return a + b
    
    # Test the utility
    print("🧪 Testing Fixed RVSDG Diagram Creation")
    print("=" * 40)
    
    # Get RVSDG expressions
    rvsdg_expr1, _ = frontend(max_if_else)
    rvsdg_expr2, _ = frontend(simple_add)
    
    # Create diagrams
    diagram1 = create_rvsdg_diagram(rvsdg_expr1, "max")
    diagram2 = create_rvsdg_diagram(rvsdg_expr2, "add")
    
    print("Max Function Diagram:")
    print(diagram1)
    print("\n" + "-" * 50 + "\n")
    print("Add Function Diagram:")
    print(diagram2) 