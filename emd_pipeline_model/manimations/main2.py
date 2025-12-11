from manim import *
import numpy as np

# Helper function to generate dummy signal data
def get_noisy_signal(t):
    return 0.5 * np.sin(20 * t) + 0.3 * np.sin(5 * t) + 0.2 * np.cos(t) + 0.1 * np.random.normal(0, 1, len(t))

class FullPipelineSequence(Scene):
    def construct(self):
        self.part_1_emd_processing()
        self.clear()
        self.part_2_feature_extraction()
        self.clear()
        self.part_3_training_evaluation()
        self.clear()
        # You can re-add part 3 if needed

    # ========================================================
    # PART 1: PREPROCESSING (EMD & RECONSTRUCTION)
    # ========================================================
    def part_1_emd_processing(self):
        # Title
        title = Text("Stage 1: Signal Preprocessing", font_size=40, color=BLUE).to_edge(UP)
        subtitle = Text("EMD Decomposition & Filtering", font_size=24, color=GRAY).next_to(title, DOWN)
        self.play(Write(title), FadeIn(subtitle))

        # Axes
        axes = Axes(
            x_range=[0, 10],
            y_range=[-2, 2],
            x_length=8,
            y_length=3
        ).shift(UP * 0.5)

        # Manual axis labels (Text instead of LaTeX)
        x_label = Text("Time", font_size=20).next_to(axes.x_axis, DOWN)
        y_label = Text("Amplitude", font_size=20).next_to(axes.y_axis, LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Raw 3-Axis Signals
        sig_x = axes.plot(lambda x: 0.5*np.sin(5*x) + 0.2*np.random.randn(), color=RED)
        sig_y = axes.plot(lambda x: 0.5*np.cos(5*x) + 0.2*np.random.randn(), color=GREEN)
        sig_z = axes.plot(lambda x: 0.3*np.sin(3*x), color=BLUE)

        sig_label = VGroup(
            Text("X-Axis", color=RED, font_size=20),
            Text("Y-Axis", color=GREEN, font_size=20),
            Text("Z-Axis", color=BLUE, font_size=20)
        ).arrange(DOWN).next_to(axes, RIGHT)

        self.play(Create(sig_x), Create(sig_y), Create(sig_z), Write(sig_label))
        self.wait(1)

        # Magnitude formula (Replace MathTex with Text)
        formula = Text("Magnitude = sqrt(x² + y² + z²)", color=YELLOW).next_to(axes, DOWN)
        self.play(Write(formula))

        # Magnitude signal
        sig_mag = axes.plot(lambda x: 0.8*np.sin(5*x) + 0.4*np.sin(20*x), color=YELLOW)
        self.play(
            FadeOut(sig_x), FadeOut(sig_y), FadeOut(sig_z), FadeOut(sig_label),
            TransformFromCopy(formula, sig_mag)
        )
        self.play(FadeOut(formula))
        self.wait(1)

        # EMD section
        emd_group = VGroup(axes, x_label, y_label, sig_mag)
        self.remove(title,subtitle)
        self.play(emd_group.animate.scale(0.7).to_edge(UP))

        emd_text = Text("Empirical Mode Decomposition (EMD)", font_size=24).next_to(emd_group, DOWN)
        self.play(Write(emd_text))

        # IMF axes
        imf_axes_config = {"x_range":[0,10],"y_range":[-1,1],"x_length":6,"y_length":1}

        ax1 = Axes(**imf_axes_config).shift(UP * 0.5)
        imf1 = ax1.plot(lambda x: 0.4*np.sin(30*x), color=RED)
        lbl1 = Text("IMF 1 (High Freq Noise)", font_size=18, color=RED).next_to(ax1, LEFT)

        ax2 = Axes(**imf_axes_config).shift(DOWN * 1.0)
        imf2 = ax2.plot(lambda x: 0.6*np.sin(5*x), color=GREEN)
        lbl2 = Text("IMF 2..N (Main Signal)", font_size=18, color=GREEN).next_to(ax2, LEFT)

        ax3 = Axes(**imf_axes_config).shift(DOWN * 2.5)
        residue = ax3.plot(lambda x: 0.2 * x/10, color=BLUE)
        lbl3 = Text("Residue (Trend)", font_size=18, color=BLUE).next_to(ax3, LEFT)

        self.play(
            Create(ax1), Create(imf1), Write(lbl1),
            Create(ax2), Create(imf2), Write(lbl2),
            Create(ax3), Create(residue), Write(lbl3)
        )
        self.wait(2)

        # Drop IMF 1
        cross = Cross(imf1, stroke_width=4)
        drop_text = Text("Drop IMF 1", color=RED, font_size=24).next_to(ax1, RIGHT)
        self.play(Create(cross), Write(drop_text))
        self.play(FadeOut(ax1), FadeOut(imf1), FadeOut(lbl1), FadeOut(cross), FadeOut(drop_text))

        # Combine IMF2 + Residue
        plus = Text("+", font_size=40).move_to((ax2.get_center() + ax3.get_center())/2)
        self.play(Write(plus))

        final_ax = Axes(x_range=[0,10], y_range=[-2,2], x_length=8, y_length=2).move_to(DOWN * 1)
        final_sig = final_ax.plot(lambda x: 0.6*np.sin(5*x) + 0.2*x/10, color=GOLD)
        final_lbl = Text("Reconstructed Signal", color=GOLD, font_size=24).next_to(final_ax, UP)

        self.play(
            ReplacementTransform(VGroup(ax2, imf2, ax3, residue, plus), final_sig),
            Create(final_ax), Write(final_lbl),
            FadeOut(lbl2), FadeOut(lbl3), FadeOut(emd_text), FadeOut(emd_group)
        )
        self.wait(2)

        self.play(FadeOut(Group(*self.mobjects)))

    # ========================================================
    # PART 2: FEATURE EXTRACTION
    # ========================================================
    def part_2_feature_extraction(self):
        title = Text("Stage 2: Feature Extraction", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))

        ax = Axes(x_range=[0,10], y_range=[-2,2], x_length=7, y_length=2).shift(UP + LEFT*2)
        sig = ax.plot(lambda x: 0.5*np.sin(5*x) * np.exp(-0.1*x), color=GOLD)
        lbl = Text("Processed Signal", font_size=20).next_to(ax, UP)
        self.add(ax, sig, lbl)

        # Feature vector box
        vector_box = Rectangle(height=5, width=1.5, color=WHITE).shift(RIGHT * 4)
        vector_title = Text("Feature Vector", font_size=20).next_to(vector_box, UP)

        feats = ["RMS", "Skew", "Kurtosis", "Energy", "Centroid", "Rolloff"]
        feat_texts = VGroup(*[Text(f, font_size=16) for f in feats]).arrange(DOWN, buff=0.4).move_to(vector_box.get_center())

        self.play(Create(vector_box), Write(vector_title), Write(feat_texts))

        # Time-domain highlight
        brace_time = Brace(ax, DOWN)
        time_text = Text("Time Domain Analysis", font_size=20, color=YELLOW).next_to(brace_time, DOWN)
        self.play(GrowFromCenter(brace_time), Write(time_text))

        # Transform features
        dot = Dot(color=YELLOW).move_to(ax.c2p(5, 0))
        self.play(Flash(dot))
        self.play(dot.animate.move_to(feat_texts[0].get_left() + LEFT*0.2))

        self.play(Transform(feat_texts[0], Text("0.452", font_size=16, color=YELLOW).move_to(feat_texts[0])))
        self.play(FadeOut(dot))

        # Skew + Kurtosis update
        self.play(
            Transform(feat_texts[1], Text("-0.02", font_size=16, color=YELLOW).move_to(feat_texts[1])),
            Transform(feat_texts[2], Text("1.20", font_size=16, color=YELLOW).move_to(feat_texts[2]))
        )

        self.play(FadeOut(brace_time), FadeOut(time_text))


    def part_3_training_evaluation(self):
            title = Text("Stage 3: Cross-Validation & Training", font_size=40, color=BLUE).to_edge(UP)
            self.play(Write(title))

            # 1. Construct the Data Matrix
            # Create a grid of rectangles representing the dataset
            rows, cols = 8, 5
            grid = VGroup()
            for i in range(rows):
                for j in range(cols):
                    rect = Rectangle(height=0.4, width=0.8, color=WHITE, stroke_width=1)
                    if j == cols - 1: # Label column
                        rect.set_fill(color=RED if i > 3 else BLUE, opacity=0.5) # Half healthy, half faulty
                    else:
                        rect.set_fill(color=GRAY, opacity=0.2)
                    
                    rect.move_to(np.array([j - 2, 2 - i * 0.5, 0]))
                    grid.add(rect)
            
            grid.center().shift(LEFT*2)
            
            matrix_label = Text("Feature Matrix X | Labels Y", font_size=24).next_to(grid, UP)
            row_label = Text("Samples (Healthy / Faulty)", font_size=16,).next_to(grid, LEFT)
            
            self.play(FadeIn(grid), Write(matrix_label), FadeIn(row_label))
            self.wait(1)

            # 2. K-Fold Cross Validation Split
            # Highlight a "Test" block
            test_fold_rect = SurroundingRectangle(grid[0:5], color=YELLOW, buff=0.1) # First row
            test_label = Text("Test Fold", font_size=20, color=YELLOW).next_to(test_fold_rect, RIGHT)
            
            train_fold_rect = SurroundingRectangle(grid[5:], color=GREEN, buff=0.1)
            train_label = Text("Train Folds", font_size=20, color=GREEN).next_to(train_fold_rect, RIGHT)

            self.play(Create(test_fold_rect), Write(test_label))
            self.play(Create(train_fold_rect), Write(train_label))
            self.wait(1)

            # 3. Model Training Visualization
            # Show "Model" consuming Train Folds
            model_box = RoundedRectangle(height=2, width=2, color=BLUE).shift(RIGHT*4 + UP*1)
            model_txt = Text("SVM / LGBM", font_size=20).move_to(model_box)
            
            arrow_train = Arrow(train_fold_rect.get_right(), model_box.get_left(), color=GREEN)
            
            self.play(Create(model_box), Write(model_txt), GrowArrow(arrow_train))
            self.play(Indicate(model_box, color=WHITE)) # Flash to simulate learning
            
            # 4. Prediction / Evaluation
            # Show Test data going into model
            arrow_test = Arrow(test_fold_rect.get_right(), model_box.get_left(), color=YELLOW)
            self.play(ReplacementTransform(arrow_train, arrow_test))
            
            # Output Confusion Matrix
            cm_matrix = VGroup(
                Square(side_length=1).set_fill(GREEN, opacity=0.5), # TP
                Square(side_length=1).set_fill(RED, opacity=0.5),   # FP
                Square(side_length=1).set_fill(RED, opacity=0.5),   # FN
                Square(side_length=1).set_fill(GREEN, opacity=0.5)  # TN
            ).arrange_in_grid(2, 2, buff=0).shift(RIGHT*4 + DOWN*2)
            
            cm_labels = VGroup(
                Text("TP", font_size=20).move_to(cm_matrix[0]),
                Text("FP", font_size=20).move_to(cm_matrix[1]),
                Text("FN", font_size=20).move_to(cm_matrix[2]),
                Text("TN", font_size=20).move_to(cm_matrix[3])
            )
            
            arrow_pred = Arrow(model_box.get_bottom(), cm_matrix.get_top(), color=WHITE)
            
            self.play(Create(cm_matrix), Write(cm_labels), GrowArrow(arrow_pred))
            
            # 5. Final Metrics Pop-up
            score_text = Text("Accuracy: 98.5%", font_size=36, color=GOLD).next_to(cm_matrix, DOWN)
            self.play(Write(score_text))
            
            self.wait(3)