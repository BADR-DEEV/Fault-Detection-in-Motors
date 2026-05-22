from manim import *
import numpy as np

class FaultDiagnosisPipeline(Scene):
    def construct(self):
        # Configuration for professional look
        self.camera.background_color = "#1a1a1a" # Dark grey, softer than pure black
        
        # 1. Title Sequence
        title = Text("Fault Diagnosis Methodology", font_size=48, weight=BOLD).to_edge(UP)
        subtitle = Text("EMD-SVM-Q Approach", font_size=32, color=BLUE).next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)
        
        # Transition out subtitle, keep title
        self.play(FadeOut(subtitle))
        
        # ---------------------------------------------------------
        # SECTION 1: SIGNAL ACQUISITION & PREPROCESSING
        # ---------------------------------------------------------
        
        # Create a representation of a 3-axis signal
        ax = Axes(
            x_range=[0, 10, 1], y_range=[-2, 2, 1], 
            x_length=8, y_length=3,
            axis_config={"include_tip": False, "color": GREY}
        ).shift(UP * 0.5)
        
        sig_func = lambda x: 0.5 * np.sin(3*x) + 0.3 * np.cos(10*x) + 0.1 * np.random.normal()
        signal_plot = ax.plot(sig_func, color=BLUE_C)
        
        label_raw = Text("Raw Vibration (x, y, z)", font_size=24).next_to(ax, UP)
        
        self.play(Create(ax), Create(signal_plot), Write(label_raw))
        self.wait(0.5)
        
        # Formula for composite signal
        formula = MathTex(r"S(t) = \sqrt{x^2 + y^2 + z^2}").scale(0.8).next_to(ax, DOWN)
        
        self.play(Write(formula))
        self.play(signal_plot.animate.set_color(TEAL)) # Represents transformation
        
        label_comp = Text("Composite Signal S(t)", font_size=24, color=TEAL).move_to(label_raw)
        self.play(Transform(label_raw, label_comp))
        self.wait(1)
        
        # Cleanup Section 1
        group_s1 = VGroup(ax, signal_plot, formula, label_raw)
        self.play(
            group_s1.animate.scale(0.5).to_edge(LEFT).shift(UP*2),
            FadeOut(formula) # Hide formula to save space
        )
        
        # ---------------------------------------------------------
        # SECTION 2: EMD & FILTERING
        # ---------------------------------------------------------
        
        emd_title = Text("Empirical Mode Decomposition (EMD)", font_size=32, color=YELLOW).to_edge(UP).shift(DOWN*1.5)
        self.play(Write(emd_title))
        
        # Create schematic IMFs
        imf_group = VGroup()
        for i in range(4):
            # Create decreasing frequency waves
            freq = 12 - (i*3)
            amp = 0.2 + (i*0.1)
            imf_ax = Axes(x_range=[0, 10], y_range=[-1, 1], x_length=6, y_length=0.8, axis_config={"include_ticks": False})
            imf_curve = imf_ax.plot(lambda x: amp * np.sin(freq*x), color=WHITE)
            lbl = Text(f"IMF {i+1}", font_size=20).next_to(imf_ax, LEFT)
            row = VGroup(lbl, imf_ax, imf_curve)
            imf_group.add(row)
            
        imf_group.arrange(DOWN, buff=0.2).next_to(group_s1, RIGHT, buff=1).shift(DOWN*0.5)
        
        arrow_emd = Arrow(start=group_s1.get_right(), end=imf_group.get_left(), color=GREY)
        self.play(GrowArrow(arrow_emd), LaggedStart(*[Create(g) for g in imf_group], lag_ratio=0.2))
        
        # The Filtering Step: Cross out IMF 1
        cross = Cross(imf_group[0], stroke_width=4, color=RED)
        filter_text = Text("Drop IMF 1 (Noise)", font_size=24, color=RED).next_to(cross, RIGHT)
        
        self.play(Create(cross), Write(filter_text))
        self.wait(1)
        
        # Reconstruct
        brace = Brace(imf_group[1:], direction=RIGHT)
        recon_text = Text("Reconstruct\n(Signal - IMF1)", font_size=24, color=GREEN).next_to(brace, RIGHT)
        
        self.play(Create(brace), Write(recon_text))
        self.wait(1)
        
        # Cleanup Section 2
        group_s2 = VGroup(emd_title, imf_group, arrow_emd, cross, filter_text, brace, recon_text)
        self.play(FadeOut(group_s1), FadeOut(group_s2))
        
        # ---------------------------------------------------------
        # SECTION 3: FEATURE EXTRACTION
        # ---------------------------------------------------------
        
        feat_title = Text("Feature Extraction & Fusion", font_size=36).to_edge(UP).shift(DOWN)
        self.play(Write(feat_title))
        
        # Define feature blocks
        def create_feat_box(label, items, color):
            box = RoundedRectangle(width=3.5, height=2.5, corner_radius=0.2, color=color)
            title = Text(label, font_size=24, weight=BOLD, color=color).next_to(box.get_top(), DOWN, buff=0.1)
            content = VGroup(*[Text(item, font_size=18) for item in items]).arrange(DOWN, buff=0.15).next_to(title, DOWN, buff=0.2)
            return VGroup(box, title, content)
            
        box_f1 = create_feat_box("Time Domain (F1)", ["Mean", "RMS", "Skewness", "Kurtosis", "Entropy"], BLUE)
        box_f2 = create_feat_box("Freq Domain A (F2)", ["Mean Freq", "Std Freq", "Band Power", "Median Freq"], TEAL)
        box_f3 = create_feat_box("Freq Domain B (F3)", ["Spec. Centroid", "Spec. Slope", "Spec. Entropy", "Spec. Roll-off"], GREEN)
        
        feature_group = VGroup(box_f1, box_f2, box_f3).arrange(RIGHT, buff=0.5).shift(UP*0.5)
        
        self.play(LaggedStart(FadeIn(box_f1), FadeIn(box_f2), FadeIn(box_f3), lag_ratio=0.3))
        self.wait(1)
        
        # Visualize Fusion
        fusion_arrow = Arrow(start=feature_group.get_bottom(), end=feature_group.get_bottom() + DOWN*1.5, color=WHITE)
        fusion_label = Text("Feature Fusion (F4 - F7)", font_size=28).next_to(fusion_arrow, RIGHT)
        
        # A long vector representing the fused vector
        vector_rect = Rectangle(width=8, height=0.6, color=WHITE, fill_opacity=0.2, fill_color=GREY)
        vector_text = MathTex(r"[F_1, F_2, F_3] \rightarrow \text{High Dimensional Vector}").scale(0.8).move_to(vector_rect)
        vector_group = VGroup(vector_rect, vector_text).next_to(fusion_arrow, DOWN)
        
        self.play(GrowArrow(fusion_arrow), Write(fusion_label))
        self.play(Create(vector_group))
        self.wait(1)
        
        # Cleanup Section 3
        group_s3 = VGroup(feat_title, feature_group, fusion_arrow, fusion_label, vector_group)
        self.play(FadeOut(group_s3))
        
        # ---------------------------------------------------------
        # SECTION 4: CLASSIFICATION (SVM-Q) & VOTING
        # ---------------------------------------------------------
        
        # Setup Layout
        left_pane = VGroup()
        right_pane = VGroup()
        
        # Part A: SVM Visualization
        svm_title = Text("Quadratic SVM Classifier", font_size=32, color=ORANGE)
        
        # Mock decision boundary plot
        svm_ax = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=4, y_length=4, axis_config={"include_ticks":False})
        
        # Random dots
        dot_group = VGroup()
        for _ in range(15):
            dot_group.add(Dot(point=svm_ax.c2p(np.random.uniform(-1.5, -0.2), np.random.uniform(-1.5, 1.5)), color=GREEN_D, radius=0.08)) # Healthy
        for _ in range(15):
            dot_group.add(Dot(point=svm_ax.c2p(np.random.uniform(0.2, 1.5), np.random.uniform(-1.5, 1.5)), color=RED_D, radius=0.08)) # Faulty
            
        # Curved boundary
        boundary = svm_ax.plot(lambda x: 0.5 * x**2 - 0.2, color=WHITE)
        
        svm_visual = VGroup(svm_title, svm_ax, dot_group, boundary).arrange(DOWN)
        
        # Part B: Majority Voting Logic (The Paper's Core)
        vote_title = Text("Majority Voting (Per File)", font_size=32, color=BLUE)
        
        # Create a "File" container
        file_rect = Rectangle(width=5, height=3, color=WHITE)
        file_label = Text("Single File (Signal)", font_size=24).next_to(file_rect, UP)
        
        # Create Windows inside
        windows = VGroup(*[Square(side_length=0.4, fill_opacity=0.8, color=GREY) for _ in range(10)])
        windows.arrange_in_grid(rows=2, cols=5, buff=0.2).move_to(file_rect)
        
        vote_visual = VGroup(vote_title, file_label, file_rect, windows).arrange(DOWN)
        
        # Arrange main layout
        main_group = VGroup(svm_visual, vote_visual).arrange(RIGHT, buff=2).shift(DOWN*0.5)
        
        self.play(
            FadeIn(svm_visual),
            FadeIn(vote_visual)
        )
        
        # Animate Classification
        self.play(Create(boundary), run_time=1.5)
        self.wait(0.5)
        
        # Animate Windows changing color based on prediction
        # Scenario: 7 Faulty (Red), 3 Normal (Green) -> Result: Faulty
        anims = []
        pred_labels = [RED, RED, GREEN, RED, RED, GREEN, RED, RED, GREEN, RED]
        
        for w, color in zip(windows, pred_labels):
            anims.append(w.animate.set_color(color))
            
        self.play(LaggedStart(*anims, lag_ratio=0.1))
        
        # Show count
        count_text = Text("Faulty: 7  |  Healthy: 3", font_size=28).next_to(file_rect, DOWN)
        final_res = Text("DIAGNOSIS: FAULTY", font_size=36, weight=BOLD, color=RED).next_to(count_text, DOWN)
        
        self.play(Write(count_text))
        self.play(TransformFromCopy(count_text, final_res))
        self.play(Indicate(final_res))
        
        self.wait(2)
        
        # Outro
        self.play(
            FadeOut(main_group), 
            FadeOut(count_text), 
            FadeOut(final_res),
            FadeOut(title)
        )
        
        end_text = Text("Full Pipeline Implemented", font_size=40)
        self.play(Write(end_text))
        self.wait(2)