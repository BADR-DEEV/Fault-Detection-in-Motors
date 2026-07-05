from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

def create_presentation():
    prs = Presentation()

    # Define colors and styles for a professional look
    bg_color = RGBColor(240, 240, 240)  # Light grey
    title_bg_color = RGBColor(26, 26, 117)  # Dark blue
    title_text_color = RGBColor(255, 255, 255)  # White
    content_text_color = RGBColor(0, 0, 0)  # Black
    accent_color = RGBColor(0, 51, 102)  # Navy blue
    
    # Function to apply standard formatting to a slide
    def apply_standard_layout(slide, title_text, subtitle_text=""):
        title_slide_layout = prs.slide_layouts[5]  # Blank layout for custom design
        slide.shapes.title.text = title_text
        if subtitle_text:
            slide.placeholders[1].text = subtitle_text
        
        # Add background color
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = bg_color

        # Add header bar
        header_shape = slide.shapes.add_shape(
            1,  # Rectangle shape type
            Inches(0), Inches(0), Inches(10), Inches(1)
        )
        header_fill = header_shape.fill
        header_fill.solid()
        header_fill.fore_color.rgb = title_bg_color
        
        # Add title to the header bar
        title_frame = header_shape.text_frame
        title_frame.text = title_text
        title_frame.paragraphs[0].font.size = Pt(28)
        title_frame.paragraphs[0].font.bold = True
        title_frame.paragraphs[0].font.color.rgb = title_text_color
        title_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # --- TITLE SLIDE ---
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = bg_color
    
    title = slide.shapes.title
    title.text = "Physics-Guided AI for Industrial Predictive Maintenance"
    
    subtitle = slide.placeholders[1]
    subtitle.text = "PhD Defense Presentation\n\nBadr Mohammed Shehim & Marwan Hamdi Sassi\nSupervisor: Eng. Hala Elhabrush\nUniversity of Tripoli, Spring 2026"

    title.text_frame.paragraphs[0].font.size = Pt(36)
    title.text_frame.paragraphs[0].font.color.rgb = title_bg_color
    title.text_frame.paragraphs[0].font.bold = True
    subtitle.text_frame.paragraphs[0].font.size = Pt(18)
    subtitle.text_frame.paragraphs[0].font.color.rgb = content_text_color
    subtitle.left = Inches(1.5)
    subtitle.top = Inches(3)

    # --- PHASE I ---

    # Slide A: The Problem
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase I: The Baseline Challenge")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()  # Clear placeholder text

    p = text_frame.add_paragraph()
    p.text = "• Severe signal corruption & high-frequency noise"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Unstructured feature space (no driving features)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Risk of data leakage with random splits"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Need for a reproducible binary baseline"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "In Phase I, we addressed the fundamental challenge of raw vibration data quality. "
        "Our baseline dataset contained significant noise and interference. The feature space was unstructured, "
        "making it difficult to identify the key indicators of faults. A major concern was the risk of data leakage, "
        "where random train-test splits could artificially inflate performance metrics. "
        "Therefore, establishing a robust, reproducible baseline for simple binary detection (Healthy vs. Faulty) "
        "was critical before moving to more complex tasks."
    )

    # Slide B: The Solution
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase I: Adaptive Denoising & Validation")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• EMD denoising (remove IMF1 for low-freq isolation)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Vector magnitude from triaxial data"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• 2,400 samples from 220 files (50% overlap windows)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Zero-leakage: 10-fold CV, per-fold scaling"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "Our solution involved a multi-step process. We used Empirical Mode Decomposition (EMD) "
        "to adaptively denoise the signal, specifically removing the high-frequency IMF1 component "
        "that masked low-frequency fault signatures. We then computed the vector magnitude from the "
        "triaxial data to reduce dimensionality while preserving total vibration energy. "
        "To maximize our training data, we segmented the 220 files into 800-sample windows with 50% overlap, "
        "resulting in over 2,400 training samples. Crucially, we implemented a zero-data-leakage protocol "
        "using 10-fold cross-validation with per-fold standardization, ensuring that no information from "
        "the test set influenced the training process. This guarantees deterministic reproducibility."
    )

    # Slide C: Results/Validation
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase I: Proven Detection Baseline")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• Established a robust binary detection pipeline"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Identified optimal feature sets (e.g., F6)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Validated deterministic reproducibility"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Transitioned to multi-class challenges"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "The results of Phase I were a successful, validated binary detection system. "
        "We demonstrated that our preprocessing pipeline effectively cleaned the signal "
        "and that our feature engineering process could reliably distinguish between healthy and faulty states. "
        "We also identified which feature sets performed best, such as the F6 set, "
        "which combined time and frequency domain features. Most importantly, "
        "our strict validation protocol proved that our results were reproducible and not artifacts of data leakage. "
        "This solid foundation allowed us to confidently move to the more complex multi-class problem in Phase II."
    )

    # Slide D: Video Placeholder
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase I: Demonstration")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.text = "[Video of Phase I Signal Processing & EMD Denoising]"
    text_frame.paragraphs[0].font.size = Pt(24)
    text_frame.paragraphs[0].font.color.rgb = accent_color
    text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    slide.notes_slide.notes_text_frame.text = (
        "This slide serves as a placeholder for a video demonstration. "
        "The video should visually show the raw vibration signal, the EMD decomposition process, "
        "the removal of the IMF1 component, and the resulting clean signal. "
        "It reinforces the effectiveness of the adaptive denoising technique."
    )

    # --- PHASE II ---

    # Slide A: The Problem
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase II: Beyond Binary Detection")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• Binary output lacks fault-type granularity"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• 1 kHz sampling risks aliasing high-freq signatures"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• EMD complexity bottlenecks real-time edge use"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Need shift from preventive to predictive PdM"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "Phase II addressed the limitations of a simple binary classifier. "
        "Industrial maintenance requires knowing *what* kind of fault is occurring, not just *if* one exists. "
        "Our initial 1 kHz sampling rate was insufficient to capture high-frequency bearing defect signatures, "
        "risking aliasing. Furthermore, the O(N log N) complexity of EMD made it unsuitable for real-time applications "
        "on resource-limited edge devices. The goal evolved from 'is there a fault?' to 'what is the fault, "
        "and what is its predicted remaining life?' This required a more sophisticated, physics-aligned multi-class approach."
    )

    # Slide B: The Solution
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase II: Physics-Aligned Multi-Class Diagnosis")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• Used MaFaulDa dataset (50kHz -> 3846Hz)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Tachometer-based RPM estimation & tracking"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• 23 physics-engineered features (energy ratios)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• File-level GroupShuffleSplit (no leakage)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "To tackle these challenges, we moved to the MaFaulDa dataset, which offered a higher sampling rate. "
        "We decimated it to 3846 Hz, which was appropriate for capturing the targeted fault frequencies. "
        "We implemented tachometer-based RPM estimation to make our analysis aware of the machine's speed. "
        "Crucially, we designed 23 new features that were explicitly aligned with the physics of rotating machinery, "
        "such as directional vibration energy ratios and spectral shape descriptors tied to bearing fault frequencies. "
        "Finally, we enforced a strict, zero-data-leakage validation protocol using GroupShuffleSplit at the file level, "
        "ensuring that windows from the same operational run stayed together, preventing any temporal overlap "
        "between training and test sets. This is a cornerstone of physics-informed machine learning."
    )

    # Slide C: Results/Validation
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase II: Harmonic Signatures Match Physics")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• Imbalance shows 1x RPM dominant peak"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Misalignment shows 2x / 3x RPM harmonics"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Bearing faults show BPFO/BPFI harmonics"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Features encode physics, not noise"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "The validation of Phase II was rigorous and rooted in the physical reality of the system. "
        "When we analyzed the spectral features generated by our model, they perfectly matched known "
        "kinematic signatures. For example, an imbalance fault clearly showed a dominant peak at 1x RPM. "
        "Misalignment faults exhibited characteristic peaks at 2x and 3x RPM. "
        "Most importantly, bearing faults displayed the calculated Ball Pass Frequency Outer (BPFO) "
        "and Ball Pass Frequency Inner (BPFI) harmonics as predicted by the bearing geometry. "
        "This proves that our features and model learned the underlying physics, not just statistical noise, "
        "which is essential for trust in an industrial setting. Our explainable AI techniques confirmed this."
    )

    # Slide D: Video Placeholder
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase II: Demonstration")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.text = "[Video of Multi-Class Classification & XAI Insights]"
    text_frame.paragraphs[0].font.size = Pt(24)
    text_frame.paragraphs[0].font.color.rgb = accent_color
    text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    slide.notes_slide.notes_text_frame.text = (
        "This video placeholder should demonstrate the multi-class classification results. "
        "It could show a spectrogram or waterfall plot highlighting the different fault signatures "
        "as described in the previous slide. It might also include a brief visualization of the XAI analysis, "
        "like a Permutation Feature Importance chart, showing which features were most important "
        "for identifying each fault type."
    )

    # --- PHASE III ---

    # Slide A: The Problem
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase III: Limits of Standard FFT & ML")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• Variable RPM causes FFT frequency smearing"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Standard ML memorizes noise, not physics"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Manual feature engineering hits scalability ceiling"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Naive oversampling injects synthetic artifacts"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "Phase III confronted the final, most challenging barriers. "
        "Standard FFT analysis fails under variable speed conditions because the fault frequencies shift "
        "across the frequency bins, causing smearing and loss of resolution. "
        "Traditional machine learning models, even with good features, can sometimes memorize "
        "dataset-specific noise patterns instead of the universal physical laws governing machinery. "
        "Manual feature engineering, while powerful, becomes a bottleneck as the complexity of the "
        "problem increases; it doesn't scale well. Finally, common data augmentation techniques like SMOTE "
        "can introduce synthetic data points that don't conform to the underlying physical process, "
        "degrading model integrity."
    )

    # Slide B: The Solution
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase III: Order-Tracking & Spectral Deep Learning")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• Angular resampling (time -> angle domain)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Physics-constrained spectral preprocessing"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Spectral 1D-CNN for autonomous pattern recognition"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• GroupKFold isolation & spectral jitter masking"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "Our solution was a paradigm shift to a speed-invariant domain. "
        "We employed Order Tracking, converting the time-domain signal into the angular domain using "
        "the estimated RPM. This locks the fault frequencies to fixed angular positions, eliminating smearing. "
        "We then applied physics-constrained preprocessing steps like cross-axis scaling and logarithmic transforms. "
        "We designed a novel Spectral 1D-CNN architecture that operates directly on the order-tracked spectra. "
        "This allows the network to automatically learn the most relevant spectral patterns for fault diagnosis, "
        "bypassing the need for manual feature engineering. Finally, we maintained our rigorous validation "
        "standards with GroupKFold and introduced spectral-specific data augmentation techniques like "
        "jitter and masking to improve robustness without compromising physical validity."
    )

    # Slide C: Results/Validation
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase III: State-of-the-Art Performance")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• 96.92% accuracy (strictly held-out test)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Perfect precision/recall for Ball Faults"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Confusion matrix aligns with structural theory"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Dummy baseline: 13.04% (proves model learning)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "The results of Phase III were exceptional and validate our approach. "
        "The model achieved 96.92% accuracy on a strictly held-out test set, with no evidence of overfitting, "
        "thanks to our leak-proof validation. For the critical Ball Fault class, the model achieved perfect "
        "precision and recall, which is vital for safety. The confusion matrix was analyzed and found to align "
        "well with mechanical expectations; for example, vertical and horizontal misalignments showed a slight "
        "overlap, which is physically plausible, while other fault types were clearly separated. "
        "Finally, we compared our result to a dummy classifier that simply guessed the majority class, "
        "which achieved only 13.04% accuracy, proving that our model learned meaningful patterns, not chance."
    )

    # Slide D: Video Placeholder
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Phase III: Demonstration")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.text = "[Video of Order Tracking & Deep Learning Inference]"
    text_frame.paragraphs[0].font.size = Pt(24)
    text_frame.paragraphs[0].font.color.rgb = accent_color
    text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    slide.notes_slide.notes_text_frame.text = (
        "This final video placeholder should showcase the core innovation of Phase III. "
        "It could visualize the transformation from a time-domain signal with shifting peaks (under variable RPM) "
        "to a clean, order-tracked spectrum where the peaks are locked and clearly identifiable. "
        "It might also show a simplified animation of the 1D-CNN scanning the spectral bins "
        "to make its prediction."
    )

    # --- CONCLUSION & FUTURE WORK ---

    # Conclusion Slide
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Conclusion: The Evolutionary Journey")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• Phase I: Detection (Binary, Noise-Robust)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Phase II: Diagnosis (Multi-Class, Explainable)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Phase III: Automation (Deep, Speed-Invariant)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Core Insight: Physics-Constrained ML is non-negotiable"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "In conclusion, this research presented a clear evolutionary path. "
        "We started with a robust baseline for simple detection, then advanced to explainable multi-class diagnosis, "
        "and finally achieved state-of-the-art automation through deep learning in a speed-invariant domain. "
        "Throughout all phases, the central theme was the integration of physical laws into the AI pipeline. "
        "This work demonstrates that blind, purely data-driven approaches are insufficient for reliable, "
        "trustworthy industrial deployment. Physics-informed machine learning is not just beneficial; it is essential "
        "for building systems that engineers can trust and deploy with confidence."
    )

    # Future Work Slide
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    apply_standard_layout(slide, "Future Work: Towards Full Autonomy")
    left, top, width, height = Inches(1), Inches(1.5), Inches(8), Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.clear()

    p = text_frame.add_paragraph()
    p.text = "• Edge Intelligence (FPGA, Custom PCB)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Tacholess Order Tracking (RPM from harmonics)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Multimodal Fusion (Acoustics, Thermal)"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    p = text_frame.add_paragraph()
    p.text = "• Remaining Useful Life (RUL) Prognostics"
    p.font.size = Pt(20)
    p.font.color.rgb = content_text_color

    slide.notes_slide.notes_text_frame.text = (
        "Looking forward, our research roadmap includes several exciting directions. "
        "First, we aim to optimize our models for deployment on edge hardware like FPGAs or custom PCBs, "
        "bringing real-time intelligence directly to the machine. Second, we plan to develop tacholess order tracking, "
        "where the rotational speed is inferred directly from the vibration harmonics, eliminating the need for "
        "a separate encoder sensor. Third, we will explore multimodal fusion, combining vibration data with "
        "acoustic emissions and thermal imaging for a more holistic view of machine health. "
        "Finally, the ultimate goal is to extend this diagnostic capability to prognostics, "
        "predicting the Remaining Useful Life (RUL) of components, enabling truly proactive maintenance scheduling."
    )

    # Save the presentation
    filename = "Physics_Guided_AI_for_Industrial_Predictive_Maintenance_Defense.pptx"
    prs.save(filename)
    print(f"\nPresentation '{filename}' created successfully!")

if __name__ == '__main__':
    print("hello")
    create_presentation()