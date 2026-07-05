import graphviz

def visualize_report_architecture():
    # ==========================================
    # 🎨 GRAPH SETUP (REPORT-READY & HIGH-RES)
    # ==========================================
    dot = graphviz.Digraph(
        name='SpectralCNN_Chunks',
        format='png',
        graph_attr={
            'rankdir': 'LR',           # Left to Right flow
            'splines': 'line',         # Straight, rigid arrows for a blocky look
            'nodesep': '0.8',          # Vertical spacing
            'ranksep': '1.0',          # Horizontal spacing between layers
            'fontname': 'Helvetica',
            'fontsize': '20',
            'fontcolor': '#222222',
            'label': 'SpectralCNN Architecture: Feature Map Volumes\n\n',
            'labelloc': 't',           # Title at the top
            'bgcolor': '#ffffff',      # Pure white for report backgrounds
            'dpi': '300'               # 🌟 Crisp high-resolution for documents!
        },
        node_attr={
            'shape': 'box',            # 🌟 Pure rectangle ("chunk")
            'style': 'filled, solid',  # Solid fill, sharp edges (no rounded corners)
            'fontname': 'Helvetica',
            'fontsize': '12',
            'penwidth': '1.5',
            'color': '#2B2B2B',        # Dark border for contrast
        },
        edge_attr={
            'color': '#444444',
            'penwidth': '2.5',
            'arrowsize': '1.2'
        }
    )

    # ==========================================
    # 🛠️ HELPER: HTML LABELS FOR BEAUTIFUL TEXT
    # ==========================================
    # This allows us to mix Bold, Italics, and different font sizes inside the chunks.
    def make_html_label(title, layers, shape_out):
        html = '<<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="2">'
        html += f'<TR><TD><FONT POINT-SIZE="15"><B>{title}</B></FONT></TD></TR>'
        if layers:
            html += '<TR><TD> </TD></TR>' # Spacer
            for layer in layers:
                html += f'<TR><TD>{layer}</TD></TR>'
        if shape_out:
            html += '<TR><TD> </TD></TR>' # Spacer
            html += f'<TR><TD><FONT POINT-SIZE="11" COLOR="#444444"><I>{shape_out}</I></FONT></TD></TR>'
        html += '</TABLE>>'
        return html

    # ==========================================
    # 📦 DEFINE RECTANGULAR CHUNKS (CNN FUNNEL)
    # ==========================================
    # 🌟 MAGIC: We set specific `height` and `width`.
    # As the network deepens: Height decreases (Pooling), Width increases (More channels).
    
    # Input is tall and thin
    dot.node('input', 
             make_html_label('Input Signal', ['Raw Time-Series'], 'Shape: (B, 3, L)'),
             height='5.0', width='0.7', fillcolor='#F0F4F8') # Light Grey

    # Block 1 is slightly shorter, slightly wider
    dot.node('block1', 
             make_html_label('Conv Block 1', ['Conv1D (32, k=7)', 'BatchNorm + ReLU', 'MaxPool (p=2)', 'Dropout (0.1)'], 'Out: (B, 32, L/2)'),
             height='4.2', width='1.2', fillcolor='#BBDEFB') # Blue 1

    # Block 2 is shorter, wider
    dot.node('block2', 
             make_html_label('Conv Block 2', ['Conv1D (64, k=5)', 'BatchNorm + ReLU', 'MaxPool (p=2)', 'Dropout (0.2)'], 'Out: (B, 64, L/4)'),
             height='3.2', width='1.6', fillcolor='#90CAF9') # Blue 2

    # Block 3 is the shortest and widest convolutional block
    dot.node('block3', 
             make_html_label('Conv Block 3', ['Conv1D (128, k=3)', 'BatchNorm + ReLU', 'AdaptMaxPool (4)', 'Dropout (0.2)'], 'Out: (B, 128, 4)'),
             height='2.0', width='2.0', fillcolor='#64B5F6') # Blue 3

    # Flatten jumps back to being very tall and narrow to represent a 1D Vector
    dot.node('flatten', 
             make_html_label('Flatten', [], 'Out: (B, 512)'),
             height='4.8', width='0.6', fillcolor='#FFE0B2') # Light Orange

    # Dense block compresses the vector
    dot.node('fc', 
             make_html_label('Dense Block', ['Linear (128)', 'BatchNorm + ReLU', 'Dropout (0.5)'], 'Out: (B, 128)'),
             height='2.8', width='1.0', fillcolor='#FFF9C4') # Light Yellow

    # Output is the smallest vector
    dot.node('output', 
             make_html_label('Output Layer', ['Linear (6 Classes)'], 'Out: (B, 6)'),
             height='1.2', width='0.8', fillcolor='#C8E6C9') # Light Green

    # ==========================================
    # 🔗 CONNECT THE BLOCKS
    # ==========================================
    dot.edge('input', 'block1')
    dot.edge('block1', 'block2')
    dot.edge('block2', 'block3')
    dot.edge('block3', 'flatten')
    dot.edge('flatten', 'fc')
    dot.edge('fc', 'output')

    # ==========================================
    # 💾 SAVE OUTPUT
    # ==========================================
    output_name = 'SpectralCNN_Report_Architecture'
    
    dot.render(output_name, cleanup=True)
    dot.format = 'svg'
    dot.render(output_name, cleanup=True)
    
    print("✅ High-res 'Chunk' architecture generated!")
    print(f"📁 Saved as '{output_name}.png' and '.svg'")

if __name__ == "__main__":
    visualize_report_architecture()