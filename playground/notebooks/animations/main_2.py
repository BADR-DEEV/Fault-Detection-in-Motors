from manim import *
import numpy as np

class MaFaulDaPipeline(Scene):
    def construct(self):
        # ---------------------------------------------------------
        # CONFIGURATION & THEME
        # ---------------------------------------------------------
        self.camera.background_color = "#1a1a1a"
        
        # Color Palette
        C_AXIAL = "#4FC1E9"  # Cyan
        C_RADIAL = "#A0D468" # Green
        C_TAN = "#AC92EC"    # Lavender
        C_TACH = "#FFCE54"   # Yellow
        C_WIN = "#FC6E51"    # Orange
        C_TXT = "#E6E9ED"    # White
        
        # ---------------------------------------------------------
        # INTRO
        # ---------------------------------------------------------
        
        title = Text("MaFaulDa Physics-Informed Pipeline", font_size=40, color=WHITE).to_edge(UP, buff=0.5)
        subtitle = Text("Graduation Project Validation | Accuracy: 98.53%", font_size=24, color=C_TACH).next_to(title, DOWN)
        line = Line(LEFT*6, RIGHT*6, color=GRAY).next_to(subtitle, DOWN, buff=0.2)
        
        self.play(Write(title), FadeIn(subtitle), Create(line))
        self.wait(1)

        # ---------------------------------------------------------
        # SECTION 1: DATA INGESTION
        # ---------------------------------------------------------
        
        # Layout for 4 signals - SHIFTED DOWN TO AVOID TITLE COLLISION
        axes = VGroup()
        signals = VGroup()
        labels = VGroup()
        names = ["Axial", "Radial", "Tangential", "Tachometer"]
        colors = [C_AXIAL, C_RADIAL, C_TAN, C_TACH]
        
        # Adjusted Y positions to be lower
        y_positions = [1.5, 0.2, -1.1, -2.4]
        
        for i in range(4):
            ax = Axes(
                x_range=[0, 10, 1], y_range=[-2, 2, 1],
                x_length=7, y_length=1.0, # Slightly shorter height
                axis_config={"include_numbers": False, "stroke_width": 1}
            ).move_to(RIGHT * 0.5 + UP * y_positions[i])
            
            # Physics-like signals
            if i < 3: 
                freq = 15 + i*5
                # Add some random noise to simulate raw vibration
                sig = ax.plot(lambda t: 0.4*np.sin(freq*t) + 0.15*np.random.normal(), color=colors[i], stroke_width=1.2)
            else: 
                # Tachometer square wave
                sig = ax.plot(lambda t: 0.8 if (t % 1.5) < 0.2 else 0.0, color=colors[i], stroke_width=1.5)
            
            lbl = Text(names[i], font_size=16, color=colors[i]).next_to(ax, LEFT)
            axes.add(ax); signals.add(sig); labels.add(lbl)
            
        group_raw = VGroup(axes, signals, labels)
        fs_label = MathTex(r"F_s = 50 \text{ kHz}", font_size=28).next_to(group_raw, DOWN, buff=0.2)
        
        self.play(LaggedStart(*[Create(obj) for obj in axes], lag_ratio=0.1))
        self.play(LaggedStart(*[Create(obj) for obj in signals], lag_ratio=0.1), Write(labels))
        self.play(Write(fs_label))
        
        # Decimation - UPDATE TO FACTOR 13
        dec_txt = MathTex(r"\xrightarrow{\text{Decimate } \div 13}", color=C_WIN, font_size=32).next_to(fs_label, RIGHT)
        fs_new = MathTex(r"F_{new} \approx 3.8 \text{ kHz}", font_size=28, color=C_WIN).next_to(dec_txt, RIGHT)
        
        self.play(Write(dec_txt), Write(fs_new))
        self.wait(1)
        
        # ---------------------------------------------------------
        # SECTION 2: WINDOWING & EXTRACTION (CLEANER TRANSITION)
        # ---------------------------------------------------------
        
        step2_title = Text("1. Windowing & Physics Extraction", font_size=32, color=C_TXT).to_corner(UL)
        
        self.play(
            FadeOut(title), FadeOut(subtitle), FadeOut(line),
            ReplacementTransform(group_raw, step2_title), # Move data to title
            FadeOut(fs_label), FadeOut(dec_txt), FadeOut(fs_new)
        )
        
        # Signal Stream
        stream_rect = Rectangle(width=10, height=1.5, color=GRAY, fill_opacity=0.1)
        stream_wave = FunctionGraph(lambda t: 0.3*np.sin(12*t) + 0.1*np.cos(40*t), x_range=[-6, 6], color=C_AXIAL).move_to(stream_rect)
        
        # Window (UPDATED SIZE)
        window = Rectangle(width=3, height=1.7, color=C_WIN, stroke_width=4).move_to(stream_rect.get_left() + RIGHT*2)
        win_info = Tex(r"Window: 4096\\Stride: 2048", font_size=24, color=C_WIN).next_to(window, UP)
        
        stream_group = VGroup(stream_rect, stream_wave, window, win_info)
        
        self.play(Create(stream_rect), Create(stream_wave))
        self.play(Create(window), FadeIn(win_info))
        
        # Animate Slide (50% Overlap)
        self.play(window.animate.shift(RIGHT*1.5), run_time=0.6)
        
        # --- DECLUTTER STEP ---
        # Completely remove previous graphs before showing formula
        self.play(FadeOut(stream_group))
        
        # Feature Box - CENTERED
        feat_box = RoundedRectangle(height=3.5, width=7, corner_radius=0.2, color=WHITE).move_to(ORIGIN)
        feat_title = Text("Physics Feature Vector", font_size=24, color=C_WIN).next_to(feat_box, UP)
        
        feats = VGroup(
            MathTex(r"\text{RPM} = 60 / \Delta t_{tach}", color=C_TACH, font_size=28),
            MathTex(r"\text{Radial Ratio} = \frac{RMS_{rad}}{RMS_{ax} + RMS_{tan}}", color=C_RADIAL, font_size=28),
            MathTex(r"\text{Spectral Spread} = \sigma_f \text{ (Normal } > \text{ Fault)}", color=C_AXIAL, font_size=28),
            Text("(Low Spread = Tonal Fault Harmonics)", color=GRAY, font_size=20)
        ).arrange(DOWN, buff=0.4).move_to(feat_box)
        
        self.play(Create(feat_box), Write(feat_title), Write(feats))
        self.wait(2)
        
        section2_group = VGroup(feat_box, feat_title, feats)
        
        # ---------------------------------------------------------
        # SECTION 3: ABLATION (MAFAULDA REALITY CHECK)
        # ---------------------------------------------------------
        
        # step3_title = Text("2. Validation: MaFaulDa Physics (Radial Dominance)", font_size=32, color=C_TXT).to_corner(UL)
        # self.play(ReplacementTransform(step2_title, step3_title), FadeOut(section2_group))
        
        # # Chart
        # chart_ax = Axes(
        #     x_range=[0, 3, 1], y_range=[94, 100, 2],
        #     x_length=6, y_length=4,
        #     axis_config={"include_numbers": True},
        #     y_axis_config={"include_tip": False}
        # ).shift(DOWN*0.5)
        
        # y_lbl = chart_ax.get_y_axis_label(Text("Accuracy %", font_size=20).rotate(90*DEGREES), edge=LEFT, direction=LEFT, buff=0.3)
        
        # # Data from logs
        # # Full: 98.53, No Ax: 96.84 (-1.7), No Rad: 95.97 (-2.6)
        
        # bar_full = Rectangle(height=chart_ax.c2p(0, 98.53)[1] - chart_ax.c2p(0, 94)[1], width=1, color=GREEN, fill_opacity=0.8)
        # bar_full.move_to(chart_ax.c2p(0.5, 94), aligned_edge=DOWN)
        
        # bar_no_ax = Rectangle(height=chart_ax.c2p(0, 96.84)[1] - chart_ax.c2p(0, 94)[1], width=1, color=C_AXIAL, fill_opacity=0.6)
        # bar_no_ax.move_to(chart_ax.c2p(1.5, 94), aligned_edge=DOWN)
        
        # bar_no_rad = Rectangle(height=chart_ax.c2p(0, 95.97)[1] - chart_ax.c2p(0, 94)[1], width=1, color=C_RADIAL, fill_opacity=0.6)
        # bar_no_rad.move_to(chart_ax.c2p(2.5, 94), aligned_edge=DOWN)
        
        # # Labels
        # lbl_full = Text("Full", font_size=18).next_to(bar_full, DOWN)
        # val_full = Text("98.5%", font_size=20, color=GREEN).next_to(bar_full, UP)
        
        # lbl_ax = Text("No Axial", font_size=18).next_to(bar_no_ax, DOWN)
        # val_ax = Text("-1.7%", font_size=18, color=RED).next_to(bar_no_ax, UP)
        
        # lbl_rad = Text("No Radial", font_size=18).next_to(bar_no_rad, DOWN)
        # val_rad = Text("-2.6%", font_size=18, color=RED).next_to(bar_no_rad, UP)
        
        # self.play(Create(chart_ax), Write(y_lbl))
        # self.play(GrowFromEdge(bar_full, DOWN), FadeIn(lbl_full), Write(val_full))
        # self.play(GrowFromEdge(bar_no_ax, DOWN), FadeIn(lbl_ax), Write(val_ax))
        
        # # Highlight Radial Drop
        # self.play(GrowFromEdge(bar_no_rad, DOWN), FadeIn(lbl_rad), Write(val_rad))
        
        # verdict = Text("Verdict: Radial Features Critical (MaFaulDa Coupling Physics).", font_size=20, color=C_RADIAL).to_edge(DOWN)
        # self.play(Write(verdict))
        # self.wait(2)
        
        # section3_group = VGroup(chart_ax, y_lbl, bar_full, bar_no_ax, bar_no_rad, lbl_full, val_full, lbl_ax, val_ax, lbl_rad, val_rad, verdict)
        
        # ---------------------------------------------------------
        # SECTION 4: SPECTRAL SPREAD (Normal vs Fault)
        # ---------------------------------------------------------
        
        step4_title = Text("2. Bearing Physics: Spectral Spread Analysis", font_size=32, color=C_TXT).to_corner(UL)
        self.play(ReplacementTransform(step2_title, step4_title), FadeOut(section2_group))
        
        # Graph
        spec_ax = Axes(
            x_range=[0, 300, 50], y_range=[0, 1, 0.2], 
            x_length=8, y_length=4,
            axis_config={"include_numbers": True, "include_tip": False}
        ).shift(DOWN*0.5)
        
        x_lbl = Text("Frequency (Hz)", font_size=20).next_to(spec_ax, DOWN)
        self.play(Create(spec_ax), Write(x_lbl))
        
        # 1. Normal: High Spread (Broadband) ~310
        curve_norm = spec_ax.plot(lambda x: 0.2 + 0.15*np.sin(x/3)*np.cos(x/10) + 0.1*np.random.rand(), color=BLUE, stroke_opacity=0.6)
        lbl_norm = Text("Normal: High Spread (Noise)\nValue ~310", font_size=20, color=BLUE).move_to(spec_ax.c2p(150, 0.8))
        
        self.play(Create(curve_norm), Write(lbl_norm))
        self.wait(1)
        
        # 2. Fault: Low Spread (Tonal) ~143
        def fault_sig(x):
            # Sharp peaks reduce standard deviation (spread)
            y = 0.05
            y += 0.8 * np.exp(-0.05 * (x - 43)**2)  # BSF
            y += 0.6 * np.exp(-0.05 * (x - 86)**2)  # 2x
            return y
            
        curve_fault = spec_ax.plot(fault_sig, color=RED)
        lbl_fault = Text("Early Fault: Low Spread (Tonal)\nValue ~143", font_size=20, color=RED).move_to(spec_ax.c2p(150, 0.8))
        
        self.play(
            ReplacementTransform(curve_norm, curve_fault),
            ReplacementTransform(lbl_norm, lbl_fault)
        )
        
        explanation = Text("Early faults concentrate energy into harmonics -> Lower Spread.", font_size=20, color=GRAY).to_edge(DOWN * 1.5)
        self.play(Write(explanation))
        self.wait(2)
        
        section4_group = VGroup(spec_ax, x_lbl, curve_fault, lbl_fault, explanation)
        self.play(FadeOut(section4_group))

        # ---------------------------------------------------------
        # SECTION 5: FINAL METRICS (UPDATED 0% FALSE ALARM)
        # ---------------------------------------------------------
        
        final_title = Text("Final Validation Results", font_size=36, color=BLUE).to_edge(UP)
        self.play(ReplacementTransform(step4_title, final_title))
        
        # Exact values from log
        data = [
            ["Class", "Precision", "Recall", "F1 Score"],
            ["Normal", "96.0%", "100%", "0.98"],
            ["Imbalance", "100%", "99%", "0.99"],
            ["Horiz Misalign", "97%", "96%", "0.97"],
            ["Vert Misalign", "97%", "99%", "0.98"],
            ["Ball Fault", "100%", "99%", "0.99"],
            ["Outer Race", "99%", "99%", "0.99"],
        ]
        
        table = Table(
            data,
            include_outer_lines=True,
            h_buff=0.8, v_buff=0.3,
            line_config={"stroke_width": 1, "color": GRAY}
        ).scale(0.6).shift(UP*0.5)
        
        # Styling
        # Normal Row (0% False Alarm)
        row_norm = table.get_rows()[1]
        hl_norm = SurroundingRectangle(row_norm, color=GREEN, stroke_width=2)
        lbl_norm_win = Text("0% False Alarms (100% Recall)", font_size=18, color=GREEN).next_to(hl_norm, LEFT)
        
        self.play(Create(table))
        self.play(Create(hl_norm), Write(lbl_norm_win))
        
        # Summary Text
        summary = VGroup(
            Text("Overall Accuracy: 98.53%", font_size=36, color=GREEN),
            Text("Leakage-Proof | Physics-Verified | 0% False Alarms", font_size=24, color=GRAY)
        ).arrange(DOWN).next_to(table, DOWN, buff=0.5)
        
        self.play(Write(summary))
        self.wait(3)
        
        # Outro
        self.play(
            FadeOut(table), FadeOut(hl_norm), FadeOut(lbl_norm_win), 
            FadeOut(summary), FadeOut(final_title)
        )
        
        end_txt = Text("Pipeline Validated.", font_size=48).center()
        self.play(Write(end_txt))
        self.wait(2)