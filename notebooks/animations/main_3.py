from manim import *
import numpy as np

class VibrationSaliencyScene(Scene):
    def construct(self):
        # Configuration
        AXIS_NAMES = ['Axial', 'Radial', 'Tangential']
        COLORS = [BLUE, RED, GREEN]
        
        # 1. INTRO TEXT
        title = Text("Physics-Aware Saliency Map", font_size=36).to_edge(UP)
        subtitle = Text("Detecting Outer Race Fault via Gradient Attribution", font_size=24, color=GRAY).next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle))
        
        # 2. SETUP AXES (3 stacked plots)
        axes_group = VGroup()
        plots = []
        signals = []
        
        # Generate Synthetic Vibration Data (Impulses for Outer Race Fault)
        t = np.linspace(0, 4*np.pi, 200)
        # Radial has strong impulses (Physics: Load zone impact)
        sig_radial = 0.2*np.sin(5*t) + 0.5 * (np.exp(-10*(t % 1.5)) * np.cos(20*t)) 
        # Axial/Tangential have noise
        sig_axial = 0.1*np.sin(3*t) + 0.05*np.random.randn(len(t))
        sig_tangential = 0.1*np.cos(3*t) + 0.05*np.random.randn(len(t))
        data = [sig_axial, sig_radial, sig_tangential]
        
        # Create Plots
        for i, (name, sig) in enumerate(zip(AXIS_NAMES, data)):
            ax = Axes(
                x_range=[0, 12, 4], y_range=[-1, 1, 1],
                x_length=7, y_length=1.5,
                axis_config={"include_tip": False, "font_size": 16}
            ).shift(UP * (2 - i*2.2))
            
            label = Text(name, font_size=20, color=COLORS[i]).next_to(ax, LEFT)
            
            # Plot the raw signal
            plot = ax.plot_line_graph(t, sig, line_color=COLORS[i], add_vertex_dots=False, stroke_width=2)
            
            axes_group.add(ax, label, plot)
            plots.append(ax)
            signals.append(plot)

        axes_group.move_to(LEFT * 2)
        self.play(LaggedStart(*[Create(g) for g in axes_group], lag_ratio=0.1))
        
        # 3. COMPUTING SALIENCY (Visual Effect)
        # Flash the "Radial" axis because it contains the fault info
        saliency_box = SurroundingRectangle(plots[1], color=YELLOW, buff=0.1)
        calc_text = Text("Computing Gradients...", font_size=24, color=YELLOW).next_to(saliency_box, RIGHT)
        
        self.play(Create(saliency_box), Write(calc_text))
        self.wait(0.5)
        
        # 4. SHOW SALIENCY MAP (Heatmap Overlay)
        # We overlay a glowing yellow line where the impulses are
        saliency_plots = []
        for i, (ax, sig) in enumerate(zip(plots, data)):
            if i == 1: # Radial gets high saliency
                # Mask: only high amplitude areas get saliency
                saliency_sig = np.where(sig > 0.2, sig, np.nan) 
                # Create discontinuous plot for saliency
                # (Manim hack: plotting separate segments)
                s_plot = ax.plot(lambda x: 0, color=YELLOW).set_opacity(0) # Placeholder
                
                # Add highlighting rectangles for impulses
                impulses = [0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0] # approx locations
                rects = VGroup()
                for imp in impulses:
                    if imp < 12: # within graph bounds
                        rect = Rectangle(width=0.4, height=1.5, color=YELLOW, fill_opacity=0.3, stroke_width=0)
                        rect.move_to(ax.c2p(imp, 0))
                        rects.add(rect)
                
                saliency_plots.append(rects)
            else:
                saliency_plots.append(VGroup()) # Empty

        self.play(
            FadeOut(calc_text), 
            ReplacementTransform(saliency_box, saliency_plots[1])
        )

        # 5. PHYSICS VALIDATION SIDE PANEL
        panel = RoundedRectangle(width=4, height=5, corner_radius=0.2, color=WHITE)
        panel.to_edge(RIGHT)
        
        p_title = Text("Physics Validation", font_size=20, weight=BOLD).next_to(panel.get_top(), DOWN, buff=0.2)
        
        # Bar Chart for Axis Attribution
        bars = BarChart(
            values=[10, 80, 10],
            bar_names=["Ax", "Ra", "Ta"],
            y_range=[0, 100, 20],
            x_length=3, y_length=2,
            bar_colors=[BLUE, RED, GREEN]
        ).scale(0.8).next_to(p_title, DOWN, buff=0.5)
        
        dom_text = Text("Dominant: Radial", font_size=18, color=RED).next_to(bars, DOWN)
        freq_text = Text("BPFO Detected: 3.1x", font_size=18, color=YELLOW).next_to(dom_text, DOWN)

        validation_group = VGroup(panel, p_title, bars, dom_text, freq_text)
        
        self.play(Create(panel), Write(p_title))
        self.play(Create(bars))
        self.play(Write(dom_text), Write(freq_text))
        
        # 6. FINAL HIGHLIGHT
        # Draw lines from impulses on graph to the "BPFO Detected" text
        # to show the connection between time-domain saliency and frequency logic
        lines = VGroup()
        for rect in saliency_plots[1]:
            l = DashedLine(rect.get_center(), freq_text.get_left(), color=YELLOW, stroke_width=1)
            lines.add(l)
            
        self.play(Create(lines), run_time=2)
        self.wait(2)