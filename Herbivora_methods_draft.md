# Herbivora: Deep Learning Quantification of Foliar Herbivory from Digital Photographs

## Draft — Introduction, Methods, and Conclusion

*Working manuscript text. Claims about Herbivora describe the software as implemented; empirical validation placeholders are marked [TO BE COMPLETED]. In-text citations follow author–year style. Primary literature cited by Machado et al. (2016), Getman-Pickering et al. (2020), and Cornelissen et al. (2026) is included alongside those papers. Wording has been paraphrased.*

---

## 1. Introduction

Quantifying how much leaf tissue herbivores remove is central to ecology, evolutionary biology, agronomy, and pest management. Foliar area governs light capture and carbon gain; estimates of tissue loss therefore inform crop monitoring and insecticide decisions, studies of induced defense and plant chemistry, experimental defoliation, assessments of fitness under herbivory, and comparative work on plant invasions (Kessler & Baldwin, 2002; Strauss et al., 2001; War et al., 2012; Johnson, 2011; Lizaso et al., 2003; Cronin et al., 2015; Machado et al., 2016). Herbivory is widespread across biomes and plant lineages, yet its intensity varies among species, habitats, and regions (Turcotte, Thomsen, et al., 2014; Turcotte, Davies, et al., 2014; Cornelissen et al., 2026). Even relatively small amounts of tissue removal—below about 5% of leaf area—can affect plant performance, so coarse or biased scores may obscure biologically meaningful differences (Cornelissen et al., 2026). At applied scales, insect damage also carries large economic costs for agriculture and related systems (Bradshaw et al., 2016; Getman-Pickering et al., 2020). Efficient, accurate measurement of leaf-level defoliation nevertheless remains difficult (Williams & Abbott, 1991; Johnson et al., 2016).

Despite decades of use, available quantification methods still force a trade-off between speed and reliability. Visual scoring remains common because it requires little equipment and can be completed rapidly in the field (Johnson et al., 2016; Cornelissen et al., 2026; Getman-Pickering et al., 2020). Precision and accuracy of unaided estimates depend on leaf morphology, whether damage is internal or marginal, observer experience, and the grain of scoring categories; inexperienced observers, in particular, tend to inflate low damage levels (Johnson et al., 2016). In multi-species comparisons, visual scores have consistently exceeded digital benchmarks: for natural chewing damage, visual estimates were higher than image-based values in most species examined, with an average discrepancy near 9 percentage points, and for artificial hole damage the inflation reached approximately fivefold (Cornelissen et al., 2026). Visual scoring was roughly fifteen times faster than digital workflows in that study. Observer training with resources such as the Zax Herbivory Trainer can reduce both bias and time (Xirocostas et al., 2022; Cornelissen et al., 2026), yet residual error often remains largest at intermediate to high damage levels. Getman-Pickering et al. (2020) likewise found that visual estimates overstated percent herbivory relative to ImageJ and LeafByte, consistent with earlier concerns about visual overestimation at low damage (Johnson et al., 2016).

Manual grid counting—placing a fine grid beneath a leaf and tallying squares corresponding to missing tissue—can be accurate when damage is limited, but it becomes impractical as hole number or leaf size increases (Coley, 1983; Kvet & Marshall, 1971; Getman-Pickering et al., 2020; Machado et al., 2016). Dimensional approaches that predict area from length, width, or mass are widely used for intact laminae yet cannot resolve localized insect lesions (Cristofori et al., 2007; Keramatlou et al., 2015; Machado et al., 2016). Laboratory leaf-area meters (for example, LI-COR and ADC instruments) provide precise readings for intact tissue, yet they are expensive, facility-dependent, and less reliable when insects remove tissue along the margin (LI-COR, 2014; ADC, 2013; Machado et al., 2016).

Digital image analysis improved precision by converting photographs or scans into pixel counts of intact and missing tissue. Early scanner- and camera-based protocols demonstrated inexpensive leaf-area and defoliation measurement (O’Neal et al., 2002; Bradshaw et al., 2007; Igathinathane et al., 2006), and crop-specific algorithms later targeted soybean lesion area (Mura et al., 2007; Nazaré-Jr et al., 2010). ImageJ has since become a de facto standard for herbivory work (Abràmoff et al., 2004; Getman-Pickering et al., 2020), while R-based packages such as EBImage support related measurements with greater scripting automation (Pau et al., 2010; Cornelissen et al., 2026). Across a large Neotropical dataset, ImageJ and EBImage produced largely indistinguishable herbivory estimates, supporting their use when accurate, leaf-level quantification is required (Cornelissen et al., 2026). Those gains come at a cost: leaves must typically be collected or pressed, digitized, calibrated, and often corrected by hand before measurement, so processing time greatly exceeds that of visual scoring (Cornelissen et al., 2026; Getman-Pickering et al., 2020; Xirocostas et al., 2022).

Mobile applications narrowed this gap. Easy Leaf Area illustrated rapid canopy-area estimation on phones but was not designed to quantify herbivory and proved sensitive to illumination and non-green tissue (Easlon & Bloom, 2014; Machado et al., 2016). BioLeaf applies Otsu thresholding (Otsu, 1979) in CIE L\*a\*b\* color space, removes small noise components via connected-component filtering, and reconstructs damaged borders with interactive quadratic Bézier curves, yielding percent foliar loss that closely matched specialist and LI-COR estimates for soybean and related crops while remaining usable without leaf removal (Machado et al., 2016). LeafByte combines Otsu thresholding on luma, connected-component labeling (Rosenfeld & Pfaltz, 1966), planar-homography correction for camera skew (Wang et al., 2006), and optional margin redrawing; in head-to-head tests it matched ImageJ accuracy while requiring about half the analysis time, and it additionally reports absolute leaf and damage areas rather than percent loss alone (Getman-Pickering et al., 2020; Abràmoff et al., 2004). BioLeaf was faster than LeafByte in those comparisons but returned only percent herbivory (Getman-Pickering et al., 2020). Both apps still rest primarily on classical computer-vision assumptions—strong foreground–background contrast and user adjustment of thresholds or contours when margins are incomplete (Sezgin & Sankur, 2004; Otsu, 1979)—and share limitations with desktop tools when shadows, variegation, highly ruffled or overlapping leaflets, or non-chewing damage dominate the image (Getman-Pickering et al., 2020).

Deep-learning estimators have recently been proposed for large image sets. HerbiEstim, for example, uses generative models to reconstruct damaged leaves and then quantifies area loss (Wang et al., 2024). Comparative evaluation indicates that such tools can underestimate herbivory relative to ImageJ when damage lies on the margin and may fail to return usable readings for a nontrivial fraction of images (about one-fifth of species in one multi-method evaluation) (Cornelissen et al., 2026; Wang et al., 2024). Cornelissen et al. (2026) therefore recommend digital image analysis when the scientific goal is accurate comparison among leaves or plants with similar damage levels, while reserving trained visual estimation for broad patterns that demand very large samples (see also Johnson et al., 2016; Xirocostas et al., 2022). That guidance leaves open a practical need for desktop software that (i) isolates leaves from heterogeneous backgrounds, (ii) reconstructs incomplete silhouettes when margins are consumed, (iii) separates herbivory from visually related classes such as frass, and (iv) retains interactive audit and correction before metrics are exported.

Here we describe Herbivora, a desktop graphical application for quantifying foliar herbivory from digital photographs (Sandoval, 2026). Herbivora implements a three-stage deep-learning pipeline: leaf isolation with a hybrid BiRefNet and MobileSAM model (with classical and interactive alternatives), mask-to-mask contour completion with a U-Net, and multiclass damage segmentation with a second U-Net, followed by pixel-area metrics and optional absolute scaling via a circular reference marker. Interactive editors allow users to refine both the reconstructed leaf region of interest (ROI) and the damage mask. The software targets noncommercial research and education, runs on Windows, macOS, and Linux with optional GPU acceleration, and exports tabular results for downstream analysis. The following sections detail architecture, image-processing steps, quantitative definitions of leaf area and herbivory, and a recommended protocol for reproducible use.

---

## 2. Methods

### 2.1 Software overview

Herbivora (version 1.3.11 at the time of this draft) is a Python desktop application with a CustomTkinter graphical user interface (Sandoval, 2026). It is distributed as platform installers and as source code for Windows, macOS, and Linux. The interface organizes work into a Project tab (input/output folders and installation checks) and three ordered analysis stages: Segmentation, Contour/ROI, and Analysis. Trained Herbivora checkpoints for contour completion and damage segmentation are obtained from the Hugging Face repository `mariosandovalmx/Herbivora`. Third-party components used at inference include MobileSAM weights (Ultralytics assets) and BiRefNet_lite (`ZhengPeng7/BiRefNet_lite`). Inference uses NVIDIA CUDA when available on Windows or Linux, Apple Metal Performance Shaders (MPS) when available on macOS for supported stages, and CPU otherwise. Approximate disk requirements for the environment and weights are 3–6 GB.

Herbivora is designed for batch processing of leaf photographs in a user-defined input folder. Outputs are written to a structured project directory containing intermediate white-background composites, binary leaf masks, contour previews, analyzed overlays, and a comma-separated results table (`results.csv`). The canonical pipeline canvas size is 1024 × 1024 pixels with letterboxing to preserve aspect ratio.

### 2.2 Recommended image acquisition

Measurement quality depends strongly on how leaves are imaged, a point emphasized across digital herbivory protocols (O’Neal et al., 2002; Bradshaw et al., 2007; Cornelissen et al., 2026; Getman-Pickering et al., 2020; Machado et al., 2016). We recommend practices consistent with those guidelines.

Leaves should be photographed or scanned against a background that contrasts strongly with the lamina (typically light for green tissue; dark for pale tissue when classical intact-leaf mode is used). Shadows, glare, and overlapping leaflets should be minimized. Cornelissen et al. (2026) found little difference in accuracy between color and grayscale images or between photographs and flatbed scans, yet they advise retaining color (fewer preprocessing steps; easier detection of shadows and blemishes) and preferring scanned, pressed leaves when feasible because scanners reduce scale and perspective error (see also Bradshaw et al., 2007). Getman-Pickering et al. (2020) recommend a lightbox when shadows otherwise compromise background separation, and they caution that camera tilt beyond about 15° (and ideally not beyond 30° even with skew correction based on planar geometry; Wang et al., 2006) elevates area error. Machado et al. (2016) similarly stressed a contrasting portable background for non-destructive field photography.

When absolute area in square centimeters is required, each photograph should include a circular blue reference marker of known diameter in the same plane as the leaf (Herbivora default diameter: 6.0 mm). For method or species comparisons, illumination, working distance, and leaf layout should be standardized. Multiple leaves per image can reduce measurement time without sacrificing accuracy in digital workflows (Cornelissen et al., 2026), provided individuals remain unambiguously separable.

Herbivora is intended primarily for chewing damage that creates holes or clear lamina loss relative to a reconstructed silhouette. Consistent with limitations reported for LeafByte, ImageJ, and BioLeaf, piercing–sucking and galling damage, highly ruffled or strongly overlapping compound leaves, and leaves that are almost entirely consumed remain difficult to score from a single damaged image; before-and-after photography is preferable when designs allow (Getman-Pickering et al., 2020). Digitized herbarium specimens may also be analyzable when image quality is adequate, extending applications toward historical and global-change questions (Meineke et al., 2018; Getman-Pickering et al., 2020).

### 2.3 Overall analytical workflow

For each leaf image, Herbivora executes three stages.

First, the leaf is segmented from the background and composited onto a standardized white canvas, producing a companion binary mask. Optional blue-dot scale detection stores geometric metadata for later conversion of pixels to square centimeters. Second, a contour-completion model predicts a filled leaf silhouette (ROI) from the partial mask, recovering enclosed gaps and, when possible, missing margin segments—addressing a failure mode repeatedly noted for visual scoring and for some automated deep-learning tools when damage is concentrated on the edge (Johnson et al., 2016; Cornelissen et al., 2026; Wang et al., 2024). Users may interactively edit the contour, analogous in purpose to Bézier or freehand margin reconstruction in BioLeaf and LeafByte (Getman-Pickering et al., 2020; Machado et al., 2016), but applied after a learned silhouette prediction. Third, a damage segmentation model classifies pixels inside the ROI, and Herbivora aggregates predicted damage with geometric hole detection to compute percent herbivory and optional absolute areas. Users may interactively revise the damage mask before export.

Stages are modular so that intermediates can be inspected, re-run, or corrected without repeating the full pipeline.

### 2.4 Stage 1: Leaf segmentation

#### 2.4.1 BiRefNet_lite and MobileSAM (recommended)

The default segmentation method combines BiRefNet_lite, a transformer-based foreground/background model, with MobileSAM, a lightweight promptable segment-anything model. Images are letterboxed to the pipeline resolution. BiRefNet produces a soft leaf probability map that is thresholded to a binary mask. MobileSAM is prompted with spatial priors derived from the BiRefNet prediction and returns a complementary mask. The two masks are merged under a configurable hybrid rule. In the default `birefnet_primary` mode, BiRefNet edge detail is retained within a dilated MobileSAM support region. Alternative merge modes include MobileSAM-primary, intersection, and union. Agreement can be monitored with an intersection-over-union (IoU) threshold (default 0.85); when agreement is low, fallback strategies (BiRefNet alone, SAM center prompt, or a five-point grid) are applied.

The segmented leaf is pasted onto a white canvas at 1024 × 1024 pixels, and the binary mask is stored for contour inference. Optional gray-world white balancing may be applied upstream of scale detection.

#### 2.4.2 Classical intact-leaf segmentation

As a fast classical alternative for intact or near-intact leaves on high-contrast backgrounds, Herbivora provides an Otsu-based pathway on the L channel of CIE L\*a\*b\* space, with optional HSV saturation filtering. This option follows the same global-thresholding principle used in BioLeaf (Otsu in L\*a\*b\*) and LeafByte (Otsu on luma with optional manual override) (Otsu, 1979; Getman-Pickering et al., 2020; Machado et al., 2016; Sezgin & Sankur, 2004), but it is not the default pathway for damaged leaves against complex backgrounds.

#### 2.4.3 Interactive MobileSAM

When automatic segmentation fails (cluttered backgrounds, multiple leaves, or atypical coloration), users can place click priors in an interactive MobileSAM session and finalize corrected masks in batch while retaining the same downstream stages.

#### 2.4.4 Optional absolute scale

When enabled, Herbivora detects a blue circular reference marker using fused color and geometric cues. Given user-specified physical diameter \(d_{\mathrm{mm}}\) (default 6.0 mm), pixels are converted to square centimeters after accounting for letterbox scaling into white-background coordinates:

\[
s = \frac{\pi (d_{\mathrm{mm}}/2)^{2}}{\pi (d_{\mathrm{px}}/2)^{2}} \times 10^{-2}
\quad \text{(cm}^{2}\text{ per pixel}^{2}\text{, after geometric rescaling)}.
\]

Leaf and damage areas in square centimeters equal pixel counts multiplied by \(s\). If scale detection is disabled or fails, Herbivora reports relative (percent and pixel) metrics only. Providing absolute area in addition to percent loss addresses a limitation noted for BioLeaf, which reports percent herbivory without absolute measurements (Getman-Pickering et al., 2020; Machado et al., 2016).

### 2.5 Stage 2: Contour completion and leaf ROI

Reconstructing the original leaf outline when margins are eaten is among the most difficult steps in herbivory quantification and a documented source of bias for both observers and some AI tools (Johnson et al., 2016; Cornelissen et al., 2026; Wang et al., 2024). BioLeaf solves the problem with user-placed quadratic Bézier curves (Machado et al., 2016); LeafByte allows users to draw missing margins after automatic hole detection via connected components (Getman-Pickering et al., 2020; Rosenfeld & Pfaltz, 1966). Herbivora instead applies a dedicated mask-to-mask U-Net (Segmentation Models PyTorch; ResNet-34 encoder; single-channel input/output) that predicts a completed silhouette from a partial binary mask.

At inference, a partial mask is extracted from the white-background composite by thresholding near-white pixels (default gray threshold 240), resized to 512 × 512, normalized to \([0, 1]\), and passed through the network. The sigmoid output is binarized at 0.5 and combined with the partial mask (logical OR), optionally followed by morphology-aware refinement keyed to margin class (`smooth`, `serrated`, `lobed`, or `compound`). The tissue mask represents the visible reconstructed silhouette. For quantitative herbivory, Herbivora expands this silhouette to a filled ROI by closing internal holes (default ROI mode: `filled`), treating enclosed missing regions as part of the original lamina available to be scored as damage—consistent with the contour-filling step used in ImageJ and EBImage protocols (Abràmoff et al., 2004; Pau et al., 2010; Cornelissen et al., 2026). Users may then inspect and correct the contour with interactive drawing tools (add, remove, line, and polygon). Contour training configuration targets high validation IoU (configuration notes indicate a target exceeding 0.97); full training-set composition is reported separately [TO BE COMPLETED: species, sample size, augmentation, loss, and hold-out metrics].

### 2.6 Stage 3: Damage segmentation and herbivory metrics

#### 2.6.1 Multiclass damage U-Net

Herbivory analysis uses a second U-Net with a ResNet-34 encoder that maps an RGB leaf image to four semantic classes: background (0), damage (1), frass (2), and undamaged tissue (3). Pixels outside the leaf ROI are forced to white before inference so that non-leaf context cannot generate false positives. The RGB crop is resized to 1024 × 1024 and normalized with ImageNet channel means and standard deviations. Predictions are clipped to the ROI. Thin class-1 predictions along the contour may be filtered as edge artifacts and reassigned to undamaged tissue to reduce mask-misalignment error.

Frass is scored separately and is excluded from the headline damage percentage, allowing fecal deposits to be distinguished from removed lamina—a distinction binary thresholding pipelines do not make explicitly (Otsu, 1979; Getman-Pickering et al., 2020; Machado et al., 2016). Model training details for the damage network are [TO BE COMPLETED].

#### 2.6.2 Definition of percent herbivory

Let \(R\) denote the filled leaf ROI, \(T\) the tissue (visible silhouette) mask, and \(P\) the predicted class map. Let \(D_{\mathrm{unet}} = \{p \in R : P(p) = 1\}\). Internal holes are the set difference between the filled ROI and tissue within \(R\). White holes are bright non-foliar regions inside the ROI detected by adaptive or manual brightness thresholding on the RGB composite (default adaptive mode; manual brightness hint 235; minimum component area 3 pixels; edge band 1 pixel), excluding pixels already labeled as class 1.

Under default settings (filled ROI with marginal fill enabled), damaged pixels are

\[
D = (D_{\mathrm{unet}} \cap R) \cup H_{\mathrm{internal}} \cup H_{\mathrm{white}},
\]

and percent herbivory is

\[
\mathrm{Damage\%} = 100 \times \frac{|D|}{|R|}.
\]

When scale factor \(s\) (cm\(^{2}\) pixel\(^{-2}\)) is available,

\[
A_{\mathrm{leaf}} = |R| \times s, \qquad A_{\mathrm{damage}} = |D| \times s.
\]

Undamaged percentage is reported as \(100 - \mathrm{Damage\%}\). Ancillary outputs include visible damage percentage (U-Net class 1 only), internal-hole and white-hole contributions, frass percentage, and undamaged-class percentage. After automatic analysis, users may revise the damage mask with interactive editors; revised masks update exported metrics. This combination of learned damage labels, geometric hole detection, and optional human correction is intended to mitigate the margin-related underestimation and non-returns reported for some fully automated deep-learning estimators (Wang et al., 2024; Cornelissen et al., 2026).

### 2.7 Outputs and reproducibility

For each leaf, Herbivora writes an annotated overlay, a damage mask, a leaf-ROI mask, optional metadata sidecars, and a row in `results.csv` with identifiers, pixel counts, damage percentage, and optional square-centimeter areas. Intermediate segmentation and contour products are retained for audit. Publications should report software version, checkpoint filenames, and user-selected parameters (segmentation method, hybrid merge mode, scale diameter, white-hole settings, and whether interactive edits were applied), following the broader call for transparent, standardized herbivory protocols (Cornelissen et al., 2026; Johnson et al., 2016). Herbivora software and Herbivora-trained weights are licensed under the PolyForm Noncommercial License 1.0.0; commercial use requires prior written permission. Third-party models retain their original licenses.

### 2.8 Recommended study protocol using Herbivora

Aligning with methodological guidance from Cornelissen et al. (2026) and practical constraints documented for mobile and desktop tools (Getman-Pickering et al., 2020; Machado et al., 2016; Johnson et al., 2016), we recommend the following protocol.

(1) Define the sampling unit (leaf, leaflet, or plant) and photograph leaves under the acquisition guidelines above. (2) Organize images in a single input directory with unique filenames. (3) Create a Herbivora project, confirm model availability, and process the batch with BiRefNet + MobileSAM unless a simpler intact-leaf pathway is justified. (4) Review contour previews for leaves with margin damage; edit silhouettes when the reconstructed boundary is biologically implausible. (5) Run damage analysis with default filled-ROI settings; review overlays and edit damage masks when shadows, necrosis unrelated to herbivory, or frass are misclassified. (6) Export `results.csv` and archive the output directory. (7) When study goals require high accuracy among leaves with similar damage levels, treat Herbivora outputs as a digital estimate and, on a calibration subset, compare against an independent digital standard such as ImageJ (Abràmoff et al., 2004; Cornelissen et al., 2026; Getman-Pickering et al., 2020) [TO BE COMPLETED with concordance statistics]. (8) If visual scores are also collected, train observers (for example, with Zax; Xirocostas et al., 2022) and use visual estimation for broad screening or very large samples rather than as a substitute for digital quantification when fine differences matter (Cornelissen et al., 2026; Johnson et al., 2016).

### 2.9 [Optional empirical evaluation — TO BE COMPLETED]

[Describe datasets: species, sample sizes, natural versus artificial damage, margin versus internal damage. Compare Herbivora against ImageJ (Abràmoff et al., 2004) and, where platform constraints allow, against LeafByte or BioLeaf (Getman-Pickering et al., 2020; Machado et al., 2016). Report processing time per leaf, failure rates, bias as a function of damage intensity and margin involvement, and inter-operator repeatability with and without interactive editing, following comparison designs similar to those of Getman-Pickering et al. (2020), Johnson et al. (2016), and Cornelissen et al. (2026).]

---

## 3. Conclusion

Reliable foliar herbivory estimates remain essential for interpreting plant–herbivore interactions, yet method choice continues to trade speed against precision (Williams & Abbott, 1991; Johnson et al., 2016; Cornelissen et al., 2026). Visual estimation supports large sample sizes but systematically overstates damage relative to digital standards unless observers are carefully trained (Johnson et al., 2016; Xirocostas et al., 2022; Getman-Pickering et al., 2020; Cornelissen et al., 2026). Desktop tools such as ImageJ and EBImage provide accurate, largely interchangeable digital benchmarks when images are carefully prepared (Abràmoff et al., 2004; Pau et al., 2010; Cornelissen et al., 2026). Mobile applications such as BioLeaf and LeafByte improved accessibility and throughput while retaining classical thresholding and interactive margin repair (Otsu, 1979; Machado et al., 2016; Getman-Pickering et al., 2020). Fully automated deep-learning approaches promise further speed gains but have shown underestimation and incomplete returns when damage is concentrated on leaf margins (Wang et al., 2024; Cornelissen et al., 2026).

Herbivora occupies an intermediate niche: a desktop pipeline that separates leaf isolation, learned silhouette completion, and multiclass damage segmentation into auditable stages, while preserving interactive correction comparable in purpose to the manual border tools of BioLeaf and LeafByte (Getman-Pickering et al., 2020; Machado et al., 2016; Sandoval, 2026). Used with standardized photography and transparent reporting of version and parameters, Herbivora can support reproducible herbivory measurement in ecological and agricultural research. Future work should report formal validation against established digital benchmarks across phylogenetically diverse leaves, quantify how interactive editing reduces bias, and extend evaluation beyond chewing damage to additional damage types and biomes, consistent with the cross-method research agenda outlined by Cornelissen et al. (2026).

---

## References

Abràmoff, M. D., Magalhães, P. J., & Ram, S. J. (2004). Image processing with ImageJ. *Biophotonics International, 11*(7), 36–42.

ADC. (2013). *AM350 portable leaf area meter*. Hoddesdon, Herts, UK: ADC BioScientific.

Bradshaw, C. J. A., Leroy, B., Bellard, C., Roiz, D., Albert, C., Fournier, A., … Courchamp, F. (2016). Massive yet grossly underestimated global costs of invasive insects. *Nature Communications, 7*, 12986. https://doi.org/10.1038/ncomms12986

Bradshaw, J. D., Rice, M. E., & Hill, J. H. (2007). Digital analysis of leaf surface area: Effects of shape, resolution, and size. *Journal of the Kansas Entomological Society, 80*(4), 339–347.

Coley, P. D. (1983). Herbivory and defensive characteristics of tree species in a lowland tropical forest. *Ecological Monographs, 53*(2), 209–229. https://doi.org/10.2307/1942495

Cornelissen, T., Mendes, G. M., Silveira, F. A. O., Dáttilo, W., Guevara, R., Aguilar, R., Boaventura, M. G., Campos, R., del Val, E., Demetrio, G. R., Fagundes, M., Farias, R. de P., Fernandes, G. W., Fernandes, T., Gomes, I., Kloss, T., Kuchenbecker, J., Maracahipes, L., Neves, F., … Wetzel, W. C. (2026). Quantifying leaf herbivory: A guide to methodological trade-offs and best practices. *Ecology, 107*(2), e70308. https://doi.org/10.1002/ecy.70308

Cristofori, V., Rouphael, Y., de Gyves, E. M., & Bignami, C. (2007). A simple model for estimating leaf area of hazelnut from linear measurements. *Scientia Horticulturae, 113*(2), 221–225.

Cronin, J. T., Bhattarai, G. P., Allen, W. J., & Meyerson, L. A. (2015). Biogeography of a plant invasion: Plant–herbivore interactions. *Ecology, 96*(4), 1115–1127.

Easlon, H. M., & Bloom, A. J. (2014). Easy Leaf Area: Automated digital image analysis for rapid and accurate measurement of leaf area. *Applications in Plant Sciences, 2*(7), 1400033. https://doi.org/10.3732/apps.1400033

Getman-Pickering, Z. L., Campbell, A., Aflitto, N., Grele, A., Davis, J. K., & Ugine, T. A. (2020). LeafByte: A mobile application that measures leaf area and herbivory quickly and accurately. *Methods in Ecology and Evolution, 11*(2), 215–221. https://doi.org/10.1111/2041-210X.13340

Igathinathane, C., Prakash, V. S. S., Padma, U., Babu, G. R., & Womac, A. R. (2006). Interactive computer software development for leaf area measurement. *Computers and Electronics in Agriculture, 51*(1–2), 1–16.

Johnson, M. T. J. (2011). Evolutionary ecology of plant defences against herbivores. *Functional Ecology, 25*(2), 305–311. https://doi.org/10.1111/j.1365-2435.2011.01838.x

Johnson, M. T. J., Bertrand, J. A., & Turcotte, M. M. (2016). Precision and accuracy in quantifying herbivory. *Ecological Entomology, 41*(1), 112–121. https://doi.org/10.1111/een.12280

Keramatlou, I., Sharifani, M., Sabouri, H., Alizadeh, M., & Kamkar, B. (2015). A simple linear model for leaf area estimation in Persian walnut (*Juglans regia* L.). *Scientia Horticulturae, 184*, 36–39.

Kessler, A., & Baldwin, I. T. (2002). Plant responses to insect herbivory: The emerging molecular analysis. *Annual Review of Plant Biology, 53*, 299–328.

Kvet, J., & Marshall, J. K. (1971). Assessment of leaf area and other assimilating plant surfaces. In Z. Šesták, J. Čatský, & P. G. Jarvis (Eds.), *Plant photosynthetic production: Manual of methods* (pp. 517–555). The Hague: Junk.

LI-COR. (2014). *LI-3100C area meter*. Lincoln, NE: LI-COR Biosciences.

Lizaso, J. I., Batchelor, W. D., & Westgate, M. E. (2003). A leaf area model to simulate cultivar-specific expansion and senescence of maize leaves. *Field Crops Research, 80*(1), 1–17.

Machado, B. B., Orue, J. P. M., Arruda, M. S., Santos, C. V., Sarath, D. S., Gonçalves, W. N., Silva, G. G., Pistori, H., Roel, A. R., & Rodrigues-Jr, J. F. (2016). BioLeaf: A professional mobile application to measure foliar damage caused by insect herbivory. *Computers and Electronics in Agriculture, 129*, 44–55. https://doi.org/10.1016/j.compag.2016.09.007

Meineke, E. K., Davis, C. C., & Davies, T. J. (2018). The unrealized potential of herbaria for global change biology. *Ecological Monographs, 88*(4), 505–525. https://doi.org/10.1002/ecm.1307

Mura, W. D., Oliveira, A. L., Sgarbi, E. M., & Sachsa, L. G. (2007). Detecção automática da área foliar da soja danificada pela lagarta utilizando processamento digital de imagens. In *WUW-SIBGRAPI* (pp. 1–4). IEEE Computer Society.

Nazaré-Jr, A. C., Menotti, D., Neves, J. M. R., & Sediyama, T. (2010). Automatic detection of the damaged leaf area in digital images of soybean. In *17th International Conference on Systems, Signals and Image Processing* (pp. 449–455). IEEE Computer Society.

O’Neal, M. E., Landis, D. A., & Isaacs, R. (2002). An inexpensive, accurate method for measuring leaf area and defoliation through digital image analysis. *Journal of Economic Entomology, 95*(6), 1190–1194.

Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics, 9*(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076

Pau, G., Fuchs, F., Sklyar, O., Boutros, M., & Huber, W. (2010). EBImage—An R package for image processing with applications to cellular phenotypes. *Bioinformatics, 26*(7), 979–981. https://doi.org/10.1093/bioinformatics/btq046

Rosenfeld, A., & Pfaltz, J. L. (1966). Sequential operations in digital picture processing. *Journal of the ACM, 13*(4), 471–494. https://doi.org/10.1145/321356.321357

Sandoval, M. (2026). *Herbivora* (Version 1.3.11) [Computer software]. https://github.com/mariosandovalmx/Herbivora

Sezgin, M., & Sankur, B. (2004). Survey over image thresholding techniques and quantitative performance evaluation. *Journal of Electronic Imaging, 13*(1), 146–168.

Strauss, S. Y., Conner, J. K., & Lehtilä, K. P. (2001). Effects of foliar herbivory by insects on the fitness of *Raphanus raphanistrum*: Damage can increase male fitness. *The American Naturalist, 158*(5), 496–504.

Turcotte, M. M., Davies, T. J., Thomsen, C. J. M., & Johnson, M. T. J. (2014). Macroecological and macroevolutionary patterns of leaf herbivory across vascular plants. *Proceedings of the Royal Society B: Biological Sciences, 281*(1787), 20140555. https://doi.org/10.1098/rspb.2014.0555

Turcotte, M. M., Thomsen, C. J. M., Broadhead, G. T., Fine, P. V. A., Godfrey, R. M., Lamarre, G. P. A., … Johnson, M. T. J. (2014). Percentage leaf herbivory across vascular plant species. *Ecology, 95*(3), 788. https://doi.org/10.1890/13-1741.1

Wang, X., Klette, R., & Rosenhahn, B. (2006). Geometric and photometric correction of projected rectangular pictures. In *Proceedings of the International Conference on Image and Vision Computing New Zealand* (pp. 223–228).

Wang, Z., Jiang, Y., Diallo, A. B., & Kembel, S. W. (2024). Deep learning- and image processing-based methods for automatic estimation of leaf herbivore damage. *Methods in Ecology and Evolution, 15*(4), 732–743. https://doi.org/10.1111/2041-210X.14293

War, A. R., Paulraj, M. G., Ahmad, T., Buhroo, A. A., Hussain, B., Ignacimuthu, S., & Sharma, H. C. (2012). Mechanisms of plant defense against insect herbivores. *Plant Signaling & Behavior, 7*(10), 1306–1320.

Williams, M. R., & Abbott, I. (1991). Quantifying average defoliation using leaf-level measurements. *Ecology, 72*(4), 1510–1511. https://doi.org/10.2307/1941126

Xirocostas, Z. A., Debono, S. A., Slavich, E., & Moles, A. T. (2022). The ZAX Herbivory Trainer—Free software for training researchers to visually estimate leaf damage. *Methods in Ecology and Evolution, 13*(3), 596–602. https://doi.org/10.1111/2041-210X.13785
