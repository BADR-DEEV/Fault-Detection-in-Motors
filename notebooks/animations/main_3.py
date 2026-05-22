from manim import *
import numpy as np

class SpectralCNNPipeline(Scene):
    def construct(self):
        # =========================================================
        # CONFIGURATION & THEME
        # =========================================================
        self.camera.background_color = "#121212" # Sleek dark background
        
        # Color Palette
        C_TACH = YELLOW_D
        C_VIB = PURE_CYAN
        C_ORD = PURE_MAGENTA
        C_MASK = RED_C
        C_CONV = BLUE_D
        C_POOL = TEAL_D
        C_DROP = RED_D
        C_HL = YELLOW 
        
        # Persistent Header
        main_header = Text("Deep Learning based Motor Fault Diagnosis", font_size=32, weight=BOLD).to_edge(UP)
        sub_header = Text("Order-Tracked Spectral CNN Pipeline", font_size=20, color=LIGHT_GREY).next_to(main_header, DOWN)
        
        self.play(Write(main_header), FadeIn(sub_header))
        self.wait(1)

        # =========================================================
        # SECTION 1: DATA ACQUISITION
        # =========================================================
        header_s1 = Text("1. Data Acquisition (Time Domain)", font_size=30, weight=BOLD).to_edge(UP)
        
        # Draw Axes
        ax_raw = Axes(
            x_range=[0, 10, 1], y_range=[-2, 2, 1], 
            x_length=10, y_length=3.5,
            axis_config={"include_numbers": False, "stroke_opacity": 0.5}
        ).shift(DOWN * 0.2)
        
        # Simulate varying motor speed (Chirp signal)
        vib_func = lambda t: np.sin(t**2 / 3) * 0.8 + np.random.normal(0, 0.1)
        tach_func = lambda t: 1.5 if (t**2 / 3) % (2*np.pi) < 0.5 else 0
        
        sig_vib = ax_raw.plot(vib_func, color=C_VIB, stroke_width=2)
        sig_tach = ax_raw.plot(tach_func, color=C_TACH, stroke_width=2)
        
        # Arrange Labels
        lbl_vib = Text("Vibration (Ax, Ay, Az)", font_size=18, color=C_VIB)
        lbl_tach = Text("Tachometer Pulses", font_size=18, color=C_TACH)
        labels_group = VGroup(lbl_vib, lbl_tach).arrange(RIGHT, buff=1).next_to(ax_raw, UP)
        
        self.play(
            Transform(main_header, header_s1), FadeOut(sub_header),
            Create(ax_raw), Write(labels_group)
        )
        self.play(Create(sig_vib), run_time=1.5)
        self.play(Create(sig_tach), run_time=1.5)
        
        # Notice box
        note_box = Rectangle(width=8, height=0.8, color=C_HL, fill_opacity=0.1).to_edge(DOWN)
        note_txt = Text("Problem: Speed fluctuates, causing frequency smearing.", font_size=18).move_to(note_box)
        
        self.play(Create(note_box), Write(note_txt))
        self.wait(2)
        
        group_s1 = VGroup(ax_raw, sig_vib, sig_tach, labels_group, note_box, note_txt)

        # =========================================================
        # SECTION 2: ORDER TRACKING (THE CORE METHOD)
        # =========================================================
        header_s2 = Text("2. Order Tracking (Time → Angular Domain)", font_size=30, weight=BOLD).to_edge(UP)
        self.play(Transform(main_header, header_s2), FadeOut(group_s1))
        
        ax_time = Axes(x_range=[0, 8, 1], y_range=[-1.5, 1.5, 1], x_length=10, y_length=2).shift(UP * 1.2)
        ax_order = Axes(x_range=[0, 8, 1], y_range=[-1.5, 1.5, 1], x_length=10, y_length=2).shift(DOWN * 2.2)
        
        lbl_time_ax = Text("Time Domain (Smeared)", font_size=16).next_to(ax_time, UP, aligned_edge=LEFT)
        lbl_order_ax = Text("Order Domain (Constant Angles)", font_size=16, color=C_ORD).next_to(ax_order, DOWN, aligned_edge=LEFT)
        
        time_sig = ax_time.plot(lambda t: np.sin(t**1.5) * 0.8, color=C_VIB)
        self.play(Create(ax_time), Write(lbl_time_ax), Create(time_sig))
        
        # Uneven tach pulses
        pulse_lines = VGroup(*[DashedLine(ax_time.c2p(t, 1.5), ax_time.c2p(t, -1.5), color=C_TACH) for t in [0.5, 2.0, 3.8, 5.8, 7.9]])
        pulse_lbl = Text("Tach. Pulses (Uneven)", font_size=14, color=C_TACH).next_to(pulse_lines, RIGHT, buff=0.5)
        self.play(Create(pulse_lines), Write(pulse_lbl))
        
        self.play(Create(ax_order), Write(lbl_order_ax))
        
        # Clean order signal
        order_sig = ax_order.plot(lambda t: np.sin(2 * np.pi * t) * 0.8, color=C_ORD)
        order_lines = VGroup(*[DashedLine(ax_order.c2p(t, 1.5), ax_order.c2p(t, -1.5), color=C_TACH, stroke_opacity=0.5) for t in [1.6, 3.2, 4.8, 6.4, 8.0]])
        
        arrow_resample = Arrow(ax_time.get_bottom(), ax_order.get_top(), color=WHITE)
        resample_text = Text("Resample: 64 Orders/Rev × 8 Revs = 512 points", font_size=14).next_to(arrow_resample, RIGHT)

        self.play(Create(arrow_resample), Write(resample_text))
        self.play(
            ReplacementTransform(time_sig.copy(), order_sig),
            ReplacementTransform(pulse_lines.copy(), order_lines),
            run_time=2
        )
        self.wait(2)
        
        group_s2 = VGroup(ax_time, lbl_time_ax, time_sig, pulse_lines, pulse_lbl, ax_order, lbl_order_ax, order_sig, order_lines, arrow_resample, resample_text)

        # =========================================================
        # SECTION 3: FFT & LOG SCALING
        # =========================================================
        header_s3 = Text("3. Order Spectrum Extraction (FFT)", font_size=30, weight=BOLD).to_edge(UP)
        self.play(Transform(main_header, header_s3), FadeOut(group_s2))
        
        # 🚨 FIX: Shifted ax_fft UP slightly and reduced y_length to prevent overlap with the text later
        ax_fft = Axes(
            x_range=[0, 32, 4], y_range=[0, 1.2, 0.5], 
            x_length=10, y_length=3.2,
            axis_config={"include_numbers": True}
        ).shift(UP * 0.1)
        
        fft_x_label = Text("Frequency (Orders)", font_size=16).next_to(ax_fft.x_axis, DOWN)
        fft_y_label = Text("Log Amplitude", font_size=16).next_to(ax_fft.y_axis, UP, aligned_edge=LEFT)
        
        def spectrum_func(x):
            base = 0.05 * np.random.rand()
            p1 = 0.8 * np.exp(-((x-1)**2)/0.1)
            p2 = 0.6 * np.exp(-((x-3.5)**2)/0.1)
            p3 = 0.4 * np.exp(-((x-12)**2)/0.2)
            return base + p1 + p2 + p3
            
        fft_curve = ax_fft.plot(spectrum_func, color=C_ORD, stroke_width=3)
        fft_area = ax_fft.get_area(fft_curve, color=C_ORD, opacity=0.3)
        
        hanning_txt = Text("Hanning Window + rFFT + Log1p Normalization", font_size=18, color=C_HL).to_edge(UP).shift(DOWN*1.2)
        
        self.play(Create(ax_fft), Write(fft_x_label), Write(fft_y_label), FadeIn(hanning_txt))
        self.play(Create(fft_curve), FadeIn(fft_area), run_time=1.5)
        
        fault_arrow = Arrow(ax_fft.c2p(5.5, 0.8), ax_fft.c2p(3.5, 0.6), color=YELLOW, buff=0)
        fault_lbl = Text("Fault Harmonic", font_size=16, color=YELLOW).next_to(fault_arrow.get_start(), RIGHT)
        
        self.play(Create(fault_arrow), Write(fault_lbl))
        self.wait(1.5)
        
        group_s3 = VGroup(ax_fft, fft_x_label, fft_y_label, fft_curve, fft_area, hanning_txt, fault_arrow, fault_lbl)

        # =========================================================
        # SECTION 4: DATA AUGMENTATION (SPECTRAL MASKING)
        # =========================================================
        header_s4 = Text("4. Data Augmentation (Spectral Masking)", font_size=30, weight=BOLD).to_edge(UP)
        self.play(Transform(main_header, header_s4), FadeOut(hanning_txt), FadeOut(fault_arrow), FadeOut(fault_lbl))
        
        mask_rect = Rectangle(width=1.5, height=3.2, color=C_MASK, fill_opacity=0.4).move_to(ax_fft.c2p(12, 0.6))
        mask_txt = Text("Random Frequency Bins Zeroed", font_size=16, color=C_MASK).next_to(mask_rect, UP)
        
        def masked_spectrum(x):
            if 10 < x < 14: return 0.05 * np.random.rand()
            return spectrum_func(x)
            
        masked_curve = ax_fft.plot(masked_spectrum, color=C_ORD, stroke_width=3)
        masked_area = ax_fft.get_area(masked_curve, color=C_ORD, opacity=0.3)
        
        self.play(Create(mask_rect), Write(mask_txt))
        self.play(ReplacementTransform(fft_curve, masked_curve), ReplacementTransform(fft_area, masked_area))
        
        # 🚨 FIX: Explicitly locked to the bottom edge with a buffer, far away from the x-axis numbers
        note_aug = Text("Forces CNN to learn distributed features, preventing overfitting", font_size=18, color=YELLOW, slant=ITALIC).to_edge(DOWN, buff=0.5)
        self.play(Write(note_aug))
        self.wait(2)
        
        group_s4 = VGroup(group_s3, mask_rect, mask_txt, masked_curve, masked_area, note_aug)

        # =========================================================
        # SECTION 5: 1D CNN ARCHITECTURE
        # =========================================================
        header_s5 = Text("5. 1D CNN Architecture (3 Conv Blocks)", font_size=30, weight=BOLD).to_edge(UP)
        self.play(Transform(main_header, header_s5), FadeOut(group_s4))
        
        def create_block(text, color, w, h):
            rect = RoundedRectangle(corner_radius=0.1, width=w, height=h, color=color, fill_opacity=0.6)
            lbl = Text(text, font_size=14, line_spacing=0.8).move_to(rect)
            return VGroup(rect, lbl)

        b_in = create_block("Input\n3x512", C_ORD, 1.2, 1.2)
        b_c1 = create_block("Conv1D(32)\nMaxPool(2)\nDrop(0.1)", C_CONV, 1.6, 1.4)
        b_c2 = create_block("Conv1D(64)\nMaxPool(2)\nDrop(0.2)", C_CONV, 1.6, 1.4)
        b_c3 = create_block("Conv1D(128)\nAdaptPool(4)\nDrop(0.2)", C_CONV, 1.8, 1.4)
        b_fc = create_block("Dense(128)\nDrop(0.5)", C_POOL, 1.4, 1.2)
        b_out = create_block("Output\n(6)", GREEN_D, 1.0, 1.2)
        
        blocks = VGroup(b_in, b_c1, b_c2, b_c3, b_fc, b_out)
        blocks.arrange(RIGHT, buff=0.5)
        blocks.scale_to_fit_width(13)
        blocks.move_to(ORIGIN)

        arrows = VGroup(*[Arrow(blocks[i].get_right(), blocks[i+1].get_left(), buff=0.1, max_tip_length_to_length_ratio=0.15) for i in range(len(blocks)-1)])
        
        self.play(FadeIn(b_in))
        for i in range(len(arrows)):
            self.play(GrowArrow(arrows[i]), FadeIn(blocks[i+1]), run_time=0.6)
        
        classes_str = "Classes: Normal, Imbalance, H-Misalign, V-Misalign, Ball Fault, Outer Race"
        class_labels = Text(classes_str, font_size=16, color=WHITE).next_to(blocks, DOWN, buff=1)
        self.play(Write(class_labels))
        self.wait(2)

        net_group = VGroup(blocks, arrows, class_labels)

        # =========================================================
        # SECTION 6: TRAINING & RESULTS (UPDATED WITH REAL METRICS)
        # =========================================================
        header_s6 = Text("6. Phase III: CNN Experimental Validation", font_size=30, weight=BOLD).to_edge(UP)
        self.play(Transform(main_header, header_s6), FadeOut(net_group))
        
        leak_note = Text("Evaluated via strict GroupShuffleSplit to prove Leakage-Free Physics Learning", font_size=18, color=LIGHT_GREY).next_to(main_header, DOWN)
        self.play(Write(leak_note))
        
        # 🚨 FIX: Updated with exact findings from Phase III text
        col_labels = [
            Text("Cross-Validation (5-Fold)", font_size=18, weight=BOLD),
            Text("Held-Out Test Set", font_size=18, weight=BOLD)
        ]
        
        row_labels = [
            Text("Overall Accuracy", font_size=18, weight=BOLD),
            Text("Ball Fault (F1-Score)", font_size=18, weight=BOLD)
        ]
        
        data = [
            [Text("95.74%", font_size=18), Text("96.92%", font_size=18, color=GREEN)],
            [Text("1.00 (Perfect)", font_size=18), Text("1.00 (Perfect)", font_size=18, color=GREEN)]
        ]
        
        table = MobjectTable(
            data,
            col_labels=col_labels,
            row_labels=row_labels,
            top_left_entry=Text("Metric", font_size=18, color=YELLOW),
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": GRAY}
        )
        table.scale_to_fit_width(11).shift(UP * 0.2) 
        
        # Color code the header row
        for i in range(1, 4): 
            table.get_cell((1, i)).set_color(YELLOW)
            
        # Highlight the test set column
        test_col_rect = SurroundingRectangle(
            VGroup(*[table.get_cell((r, 3)) for r in range(1, 4)]), # Rows 1, 2, 3 in Col 3
            color=GREEN, stroke_width=4, buff=0.1
        )
        
        # Baseline note
        baseline_box = Rectangle(width=10, height=0.6, color=BLUE_D, fill_opacity=0.2).next_to(table, DOWN, buff=0.5)
        baseline_note = Text("Baseline Competency Spectral CNN (96.92%)", font_size=18, color=WHITE).move_to(baseline_box)

        self.play(Create(table))
        self.wait(1)
        self.play(Create(test_col_rect))
        self.play(Create(baseline_box), Write(baseline_note))
        self.wait(4)
        
        # Outro
        self.play(FadeOut(table), FadeOut(test_col_rect), FadeOut(baseline_box), FadeOut(baseline_note), FadeOut(leak_note), FadeOut(main_header))
        
        final_text = Text("Spectral 1D-CNN Pipeline Complete", font_size=32, color=YELLOW)
        final_sub = Text("Successfully isolates high-frequency micro-transients from ambient noise", font_size=20).next_to(final_text, DOWN)
        
        self.play(Write(final_text), FadeIn(final_sub))
        self.wait(3)