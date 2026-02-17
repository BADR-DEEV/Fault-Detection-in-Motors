from manim import *
import numpy as np

class MotorFaultPipeline(Scene):
    def construct(self):
        # ---------------------------------------------------------
        # CONFIGURATION & THEME
        # ---------------------------------------------------------
        self.camera.background_color = "#1e1e1e"
        
        # Colors
        C_SIG_RAW = BLUE
        C_SIG_SMV = TEAL
        C_WIN = YELLOW
        C_NOISE = RED_C
        C_CLEAN = GREEN
        C_FEAT_T = BLUE_D   # Temporal
        C_FEAT_A = GOLD_D   # Freq A
        C_FEAT_B = MAROON_D # Freq B
        
        # Main Header (Persistent)
        main_header = Text("Motor Fault Diagnosis Pipeline", font_size=36).to_edge(UP)
        sub_header = Text("Sensors 2021 | Methodology Visualization", font_size=24, color=GRAY).next_to(main_header, DOWN)
        
        self.play(Write(main_header), FadeIn(sub_header))
        self.wait(1)

        # ---------------------------------------------------------
        # SECTION 1: DATA ACQUISITION & SMV
        # ---------------------------------------------------------
        
        # 1. Setup Axes
        axes = Axes(
            x_range=[0, 10, 1], y_range=[-2, 2, 1], 
            x_length=9, y_length=3,
            axis_config={"include_numbers": False, "stroke_opacity": 0.5}
        ).shift(UP * 0.5)
        
        # 1.2 Generate 3-Axis Signals
        x_func = lambda t: 0.4 * np.sin(3*t) + 0.05 * np.random.normal()
        y_func = lambda t: 0.4 * np.cos(3.5*t) + 0.05 * np.random.normal()
        z_func = lambda t: 0.2 * np.sin(5*t) + 0.05 * np.random.normal()
        
        sig_x = axes.plot(x_func, color=BLUE, stroke_width=2)
        sig_y = axes.plot(y_func, color=GREEN, stroke_width=2)
        sig_z = axes.plot(z_func, color=RED, stroke_width=2)
        
        labels_3axis = VGroup(
            MathTex("A_x", color=BLUE), MathTex("A_y", color=GREEN), MathTex("A_z", color=RED)
        ).arrange(DOWN).next_to(axes, LEFT)

        self.play(FadeIn(axes), Write(labels_3axis))
        self.play(Create(sig_x), Create(sig_y), Create(sig_z), run_time=1.5)
        
        # 1.3 Calculate SMV
        smv_formula = MathTex(r"S(t) = \sqrt{A_x^2 + A_y^2 + A_z^2}", font_size=32).next_to(axes, DOWN, buff=0.5)
        self.play(Write(smv_formula))
        
        # Combined signal function
        # Centered visually (no offset)
        smv_func = lambda t: np.sqrt(x_func(t)**2 + y_func(t)**2 + z_func(t)**2) - 0.5 
        sig_smv = axes.plot(smv_func, color=C_SIG_SMV, stroke_width=3)
        
        self.play(
            ReplacementTransform(VGroup(sig_x, sig_y, sig_z), sig_smv),
            FadeOut(labels_3axis),
            smv_formula.animate.scale(0.8).to_corner(DR)
        )
        self.wait(0.5)

        group_s1 = VGroup(axes, sig_smv, smv_formula)

        # ---------------------------------------------------------
        # SECTION 2: SEGMENTATION
        # ---------------------------------------------------------
        
        header_s2 = Text("1. Segmentation (Windowing)", font_size=36).to_edge(UP)
        self.play(
            Transform(main_header, header_s2),
            FadeOut(sub_header),
            group_s1.animate.scale(0.7).to_edge(LEFT).set_opacity(0.3)
        )

        strip = Rectangle(width=8, height=1, color=GRAY, fill_opacity=0.2)
        strip_wave = FunctionGraph(lambda x: 0.3*np.sin(10*x)*np.cos(x), x_range=[-4, 4], color=C_SIG_SMV).move_to(strip)
        strip_group = VGroup(strip, strip_wave).shift(UP*0.5)
        
        window_box = Rectangle(width=2, height=1.2, color=C_WIN, stroke_width=4).move_to(strip.get_left() + RIGHT)
        win_label = Tex("L=800", font_size=24, color=C_WIN).next_to(window_box, UP)
        
        self.play(FadeIn(strip_group), Create(window_box), FadeIn(win_label))
        self.play(window_box.animate.shift(RIGHT*1.5), run_time=0.6)
        self.play(window_box.animate.shift(RIGHT*1.5), run_time=0.6)
        
        extracted_sig = strip_wave.copy().set_color(C_SIG_SMV)
        self.play(
            FadeOut(strip_group), FadeOut(group_s1), FadeOut(win_label),
            window_box.animate.scale(3).move_to(ORIGIN),
            extracted_sig.animate.scale(3).move_to(ORIGIN)
        )
        
        # ---------------------------------------------------------
        # SECTION 3: EMD FILTERING & RECONSTRUCTION (UPDATED)
        # ---------------------------------------------------------
        
        header_s3 = Text("2. EMD Filtering & Reconstruction", font_size=36).to_edge(UP)
        self.play(Transform(main_header, header_s3))
        
        # 1. Decomposition: Show 3 distinct lines stacked
        # IMF 1 (High freq noise)
        imf1 = FunctionGraph(lambda x: 0.1 * np.sin(60*x) * np.random.rand(), x_range=[-4, 4], color=C_NOISE).shift(UP*1.5)
        lbl1 = Tex("IMF 1 (Noise)", color=C_NOISE, font_size=24).next_to(imf1, RIGHT)
        
        # IMF 2 (Mid freq)
        imf2 = FunctionGraph(lambda x: 0.3 * np.sin(10*x), x_range=[-4, 4], color=C_CLEAN).shift(UP*0.2)
        lbl2 = Tex("IMF 2", color=C_CLEAN, font_size=24).next_to(imf2, RIGHT)
        
        # IMF 3 (Low freq)
        imf3 = FunctionGraph(lambda x: 0.3 * np.sin(3*x), x_range=[-4, 4], color=C_CLEAN).shift(DOWN*1.0)
        lbl3 = Tex("IMF 3", color=C_CLEAN, font_size=24).next_to(imf3, RIGHT)
        
        # Dots for the rest
        dots = MathTex(r"\vdots").next_to(imf3, DOWN)
        
        self.play(
            ReplacementTransform(extracted_sig, VGroup(imf1, imf2, imf3)),
            Write(lbl1), Write(lbl2), Write(lbl3), FadeIn(dots),
            FadeOut(window_box)
        )
        
        # 2. Discard Logic
        cross = Cross(imf1, stroke_width=4)
        self.play(Create(cross))
        self.wait(0.5)
        
        # Fade out noise
        self.play(
            FadeOut(imf1), FadeOut(lbl1), FadeOut(cross),
            FadeOut(dots) # Remove dots for clarity
        )
        
        # 3. Summation Animation
        # Write formula first
        sum_formula = MathTex(r"x_{rec}(t) = \sum_{i=2}^{10} \text{IMF}_i", font_size=36, color=YELLOW).to_edge(UP).shift(DOWN*1.5)
        self.play(Write(sum_formula))
        
        # Move IMF2 and IMF3 to center and morph into sum
        final_sig = FunctionGraph(lambda x: 0.3 * np.sin(10*x) + 0.3 * np.sin(3*x), x_range=[-4, 4], color=C_CLEAN).move_to(ORIGIN)
        final_lbl = Text("Clean Signal", font_size=24, color=C_CLEAN).next_to(final_sig, DOWN)

        self.play(
            # Move them together
            imf2.animate.move_to(ORIGIN),
            imf3.animate.move_to(ORIGIN),
            # Fade out labels
            FadeOut(lbl2), FadeOut(lbl3),
            run_time=1
        )
        
        # Flash/Transform into the combined signal
        self.play(
            ReplacementTransform(VGroup(imf2, imf3), final_sig),
            FadeIn(final_lbl)
        )
        self.wait(1)

        # ---------------------------------------------------------
        # SECTION 4: FEATURE EXTRACTION (DETAILED)
        # ---------------------------------------------------------
        
        header_s4 = Text("3. Feature Extraction & Fusion", font_size=36).to_edge(UP)
        self.play(
            Transform(main_header, header_s4),
            FadeOut(final_lbl), FadeOut(sum_formula)
        )
        
        # Morph signal to block
        sig_block = RoundedRectangle(height=1, width=4, corner_radius=0.2, color=C_CLEAN, fill_opacity=0.5)
        sig_txt = Text("Signal Window", font_size=20).move_to(sig_block)
        
        self.play(ReplacementTransform(final_sig, sig_block), Write(sig_txt))
        self.play(VGroup(sig_block, sig_txt).animate.shift(UP*2))
        
        # Feature Blocks
        box_T = Rectangle(height=1, width=2, color=C_FEAT_T, fill_opacity=0.6).move_to(DOWN + LEFT*3)
        txt_T = Text("Temporal\n(F1)", font_size=18).move_to(box_T)
        
        box_A = Rectangle(height=1, width=2, color=C_FEAT_A, fill_opacity=0.6).move_to(DOWN)
        txt_A = Text("Freq A\n(F2)", font_size=18).move_to(box_A)
        
        box_B = Rectangle(height=1, width=2, color=C_FEAT_B, fill_opacity=0.6).move_to(DOWN + RIGHT*3)
        txt_B = Text("Freq B\n(F3)", font_size=18).move_to(box_B)
        
        arrows = VGroup(
            Arrow(sig_block.get_bottom(), box_T.get_top(), buff=0.1),
            Arrow(sig_block.get_bottom(), box_A.get_top(), buff=0.1),
            Arrow(sig_block.get_bottom(), box_B.get_top(), buff=0.1)
        )
        
        self.play(
            Create(arrows),
            FadeIn(box_T), Write(txt_T),
            FadeIn(box_A), Write(txt_A),
            FadeIn(box_B), Write(txt_B)
        )
        
        # Show F4 Composition
        brace_f4 = Brace(VGroup(box_T, box_A), DOWN)
        lbl_f4 = Text("F4 = T + Fa", font_size=20, color=YELLOW).next_to(brace_f4, DOWN)
        self.play(Create(brace_f4), Write(lbl_f4))
        self.wait(0.5)
        self.play(FadeOut(brace_f4), FadeOut(lbl_f4))
        
        # Show F6 Composition (Best)
        brace_f6 = Brace(VGroup(box_T, box_B), DOWN, buff=0.5)
        lbl_f6 = Text("F6 = Temporal + Freq B", font_size=24, weight=BOLD, color=GREEN).next_to(brace_f6, DOWN)
        outline_f6 = SurroundingRectangle(VGroup(box_T, box_B), color=GREEN, buff=0.1)
        
        self.play(
            Create(brace_f6), Write(lbl_f6), Create(outline_f6),
            # Dim the unused feature
            box_A.animate.set_opacity(0.2), txt_A.animate.set_opacity(0.2)
        )
        self.wait(1)
        
        feats_group = VGroup(
            sig_block, sig_txt, arrows,
            box_T, txt_T, box_A, txt_A, box_B, txt_B,
            brace_f6, lbl_f6, outline_f6
        )

        # ---------------------------------------------------------
        # SECTION 5: CLASSIFICATION (SVM-Q)
        # ---------------------------------------------------------
        
        header_s5 = Text("4. Classification: Quadratic SVM", font_size=36).to_edge(UP)
        self.play(Transform(main_header, header_s5), FadeOut(feats_group))
        
        ax_svm = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=5, y_length=4).shift(LEFT*2)
        
        np.random.seed(42)
        h_pts = VGroup(*[Dot(ax_svm.c2p(p[0], p[1]), color=BLUE, radius=0.06) for p in np.random.normal([-0.5, -0.5], 0.3, (15, 2))])
        f_pts = VGroup(*[Dot(ax_svm.c2p(p[0], p[1]), color=RED, radius=0.06) for p in np.random.normal([0.8, 0.8], 0.3, (15, 2))])
        
        boundary = ax_svm.plot(lambda x: -x + 0.3*x**2 + 0.2, x_range=[-1.8, 1.8], color=YELLOW)
        
        svm_eq = MathTex(r"K(x, y) = (\gamma x^T y + r)^2", font_size=28).next_to(ax_svm, RIGHT, buff=0.5)
        
        self.play(Create(ax_svm), Create(h_pts), Create(f_pts))
        self.play(Create(boundary), Write(svm_eq))
        self.wait(1)
        
        svm_group = VGroup(ax_svm, h_pts, f_pts, boundary, svm_eq)
        self.play(FadeOut(svm_group))

        # ---------------------------------------------------------
        # SECTION 6: RESULTS TABLE
        # ---------------------------------------------------------
        
        header_s6 = Text("Experimental Results (10-Fold CV)", font_size=36).to_edge(UP)
        self.play(Transform(main_header, header_s6))
        
        data = [
            ["Model", "Feat", "Acc", "Prec", "F1"],
            ["SVM-Q", "F6", "98.18%", "100%", "0.9818"],
            ["RF",    "F6", "97.70%", "99.2%", "0.9778"],
            ["LGBM",  "F6", "97.70%", "99.2%", "0.9778"],
            ["KNN-W", "F7", "96.84%", "99.1%", "0.9692"],
            ["LDA",   "F7", "95.01%", "100%",  "0.9501"]
        ]
        
        t0 = Table(
            data,
            include_outer_lines=True,
            h_buff=0.8, v_buff=0.4,
            line_config={"stroke_width": 1, "color": GRAY}
        ).scale(0.7)
        
        # Color Header
        for i in range(len(data[0])):
            t0.get_cell((1, i+1)).set_color(YELLOW)
            
        # Highlight Best Row
        best_row_rect = SurroundingRectangle(t0.get_rows()[1], color=GREEN, stroke_width=4)
        best_label = Text("Best Performance", font_size=24, color=GREEN).next_to(best_row_rect, RIGHT)
        
        self.play(Create(t0))
        self.play(Create(best_row_rect), Write(best_label))
        
        self.wait(3)
        self.play(FadeOut(t0), FadeOut(best_row_rect), FadeOut(best_label), FadeOut(main_header))
        
        final_text = Text("Pipeline Visualization Complete", font_size=32)
        self.play(Write(final_text))
        self.wait(2)