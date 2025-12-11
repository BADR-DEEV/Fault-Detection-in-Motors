from manim import *
import numpy as np

# ==========================================
# SCENE 1: DETAILED CONVOLUTION & POOLING
# ==========================================



class Conv1DExplainer(Scene):
    def construct(self):
        # 1. Title
        title = Text("Layer 1: Conv1D + MaxPool", font_size=36).to_edge(UP)
        self.play(Write(title))

        # 2. Input Signal Representation
        # We represent the 1024 length signal as a long line
        input_label = Text("Input Signal (1 x 1024)", font_size=24, color=BLUE).shift(UP*2)
        input_line = Line(LEFT*6, RIGHT*6, color=BLUE)
        input_group = VGroup(input_label, input_line).shift(UP*0.5)
        
        # Create a dummy waveform on top of the line
        waveform = VGroup(*[
            Line(
                input_line.point_from_proportion(i/100),
                input_line.point_from_proportion((i+1)/100) + UP * np.random.uniform(-0.1, 0.1),
                color=BLUE_A
            ) for i in range(100)
        ])
        
        self.play(FadeIn(input_group), Create(waveform))
        self.wait(1)

        # 3. The Convolution Kernel (5x1)
        kernel_width = 0.5
        kernel_box = Rectangle(width=kernel_width, height=0.4, color=YELLOW)
        kernel_label = Text("Kernel (5x1)", font_size=16, color=YELLOW).next_to(kernel_box, UP, buff=0.1)
        kernel_group = VGroup(kernel_box, kernel_label)
        
        # Position kernel at start
        kernel_group.move_to(input_line.get_left())
        
        self.play(FadeIn(kernel_group))
        
        # 4. Feature Maps (Output of Conv) - 32 Channels
        # We visualize this as a stack of lines
        feat_label = Text("Output: 32 Channels x 1024", font_size=24, color=GREEN).shift(DOWN*1)
        feat_stack = VGroup(*[
            Line(LEFT*6, RIGHT*6, color=GREEN, stroke_opacity=0.5).shift(DOWN*1 + DOWN*i*0.02)
            for i in range(10) # Show 10 lines to imply 32
        ]).center().shift(DOWN*1)
        
        self.play(FadeIn(feat_label), FadeIn(feat_stack))

        # 5. Animate Sliding Window
        # We move the kernel and reveal the feature map highlights
        scan_tracker = ValueTracker(0)
        
        kernel_group.add_updater(lambda m: m.move_to(input_line.point_from_proportion(scan_tracker.get_value())))
        
        # Add a "scanning" line on the output
        scan_line = Line(UP, DOWN, color=YELLOW, stroke_width=2).scale(0.5)
        scan_line.add_updater(lambda m: m.move_to(feat_stack.get_left() + (feat_stack.get_right() - feat_stack.get_left()) * scan_tracker.get_value()))
        
        self.add(scan_line)
        self.play(scan_tracker.animate.set_value(1), run_time=3, rate_func=linear)
        self.remove(kernel_group, scan_line)
        
        # 6. Max Pooling Explanation
        pool_label = Text("Max Pooling (Stride 2)", font_size=24, color=RED).move_to(input_label)
        
        # Transform the long green lines into shorter red lines
        pooled_stack = VGroup(*[
            Line(LEFT*3, RIGHT*3, color=RED, stroke_opacity=0.8).shift(DOWN*1 + DOWN*i*0.02)
            for i in range(10)
        ]).center().shift(DOWN*1)
        
        pooled_text = Text("Result: 32 Channels x 512", font_size=24, color=RED).next_to(pooled_stack, DOWN)

        self.play(
            Transform(title, pool_label),
            FadeOut(input_group),
            FadeOut(waveform),
            FadeOut(feat_label),
            Transform(feat_stack, pooled_stack),
            FadeIn(pooled_text)
        )
        self.wait(2)


# ==========================================
# SCENE 2: FULL ARCHITECTURE FLOW
# ==========================================
class FullArchitectureFlow(Scene):
    def construct(self):
        # Configuration
        start_pos = LEFT * 6
        
        # Helper to create layer visual
        def create_block(label, sublabel, height, width, color, pos):
            rect = Rectangle(height=height, width=width, color=color, fill_opacity=0.5)
            rect.move_to(pos)
            txt = Text(label, font_size=16).next_to(rect, UP, buff=0.1)
            sub = Text(sublabel, font_size=14, color=GRAY).next_to(rect, DOWN, buff=0.1)
            return VGroup(rect, txt, sub)

        # 1. Input
        input_block = create_block("Input", "1x1024", 0.2, 4, BLUE, start_pos + RIGHT*0.5)
        
        # 2. Layers (Scaled visually to represent shrinking length and growing depth)
        # Layer 1: 32ch, 512len
        l1 = create_block("L1: Conv+Pool", "32x512", 0.6, 2.5, GREEN, start_pos + RIGHT*3.5)
        
        # Layer 2: 64ch, 256len
        l2 = create_block("L2", "64x256", 1.0, 1.5, GREEN, start_pos + RIGHT*6)
        
        # Layer 3: 128ch, 128len
        l3 = create_block("L3", "128x128", 1.4, 0.8, GREEN, start_pos + RIGHT*8)
        
        # Layer 4-6 (Grouped for space)
        # Ends at 32ch x 64len
        l6 = create_block("L4-L6", "32x64", 0.8, 0.4, TEAL, start_pos + RIGHT*10)

        # Arrows
        arrow1 = Arrow(input_block.get_right(), l1.get_left(), buff=0.1)
        arrow2 = Arrow(l1.get_right(), l2.get_left(), buff=0.1)
        arrow3 = Arrow(l2.get_right(), l3.get_left(), buff=0.1)
        arrow4 = Arrow(l3.get_right(), l6.get_left(), buff=0.1)

        # Animation Sequence - Part 1 (Convolutions)
        self.play(Create(input_block))
        self.play(GrowArrow(arrow1), FadeIn(l1))
        self.play(GrowArrow(arrow2), FadeIn(l2))
        self.play(GrowArrow(arrow3), FadeIn(l3))
        self.play(GrowArrow(arrow4), FadeIn(l6))
        self.wait(1)

        # 3. Flattening
        # Transform the 3D block (L6) into a 1D vector
        self.play(
            FadeOut(input_block), FadeOut(l1), FadeOut(l2), FadeOut(l3),
            FadeOut(arrow1), FadeOut(arrow2), FadeOut(arrow3), FadeOut(arrow4),
            l6.animate.move_to(LEFT * 4)
        )
        
        flatten_arrow = Arrow(l6.get_right(), RIGHT*0, buff=0.5)
        flatten_text = Text("Flatten (32*64 = 2048)", font_size=20).next_to(flatten_arrow, UP)
        
        # Dense Layer 1
        fc1 = Rectangle(height=4, width=0.2, color=ORANGE, fill_opacity=0.8).move_to(RIGHT * 1)
        fc1_label = Text("FC1: 128", font_size=20).next_to(fc1, UP)
        
        # Dense Layer 2 (Output)
        fc2 = VGroup(*[Circle(radius=0.15, color=RED, fill_opacity=1) for _ in range(6)])
        fc2.arrange(DOWN, buff=0.2).move_to(RIGHT * 4)
        fc2_label = Text("Output: 6", font_size=20).next_to(fc2, UP)

        # Connections
        fc1_arrow = Arrow(l6.get_right(), fc1.get_left(), buff=0.2)
        fc2_arrow = Arrow(fc1.get_right(), fc2.get_left(), buff=0.2)

        self.play(Create(flatten_arrow), Write(flatten_text))
        self.play(TransformFromCopy(l6, fc1), FadeIn(fc1_label), Create(fc1_arrow))
        self.play(Create(fc2), FadeIn(fc2_label), Create(fc2_arrow))
        
        # 4. Final Classification Highlight
        # Highlight one class
        winner = fc2[2] # 3rd class
        winner_rect = SurroundingRectangle(winner, color=YELLOW, buff=0.1)
        winner_text = Text("Imbalance/6g", font_size=24, color=YELLOW).next_to(winner, RIGHT)
        
        self.play(Create(winner_rect), Write(winner_text))
        self.wait(2)