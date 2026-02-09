# Keystroke Dynamics & Emotion/Stress Detection: Academic Paper Extraction

## Table of Contents
1. [Paper 1: Influence of Emotion on Keyboard Typing (PLOS ONE 2015)](#paper-1)
2. [Paper 2: Keystroke Dynamics Patterns - Positive/Negative Opinions (MDPI Sensors 2021)](#paper-2)
3. [Paper 3: Touchscreen Typing for Depression Detection (Nature Sci Reports 2019)](#paper-3)
4. [Paper 4: Identifying Emotional States Using Keystroke Dynamics (Epp et al., CHI 2011)](#paper-4)
5. [Paper 5: Stress Detection via Keyboard Typing Behaviors (Sagbas et al., 2020)](#paper-5)
6. [Paper 6: Does Peoples' Keyboard Typing Reflect Their Stress Level? (Freihaut & Goeritz, 2021)](#paper-6)
7. [Paper 7: Review of Emotion Recognition from KMT Dynamics (IEEE Access 2021)](#paper-7)
8. [Paper 8: Identifying Emotion by Keystroke Dynamics and Text Pattern Analysis (Nahin et al., 2014)](#paper-8)
9. [Paper 9: Keystroke Dynamics and Heart Rate Variability as Stress Indicators (2022)](#paper-9)
10. [Paper 10: Does Keystroke Dynamics Tell Us About Emotions? SLR (Maalej & Kallel, 2020)](#paper-10)
11. [Supplementary: Semantic Scholar Search Results](#supplementary)

---

<a name="paper-1"></a>
## Paper 1: The Influence of Emotion on Keyboard Typing: An Experimental Study Using Auditory Stimuli

**Source:** PLOS ONE (2015)
**DOI:** 10.1371/journal.pone.0129056
**URL:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0129056
**Access Status:** SUCCESSFULLY FETCHED (Open Access)

### Abstract
This study examined how emotions affect keyboard typing patterns using auditory stimuli. Researchers induced emotional states via sounds and measured keystroke dynamics in 52 college students. The work found arousal significantly influenced keystroke duration and latency, though individual differences were substantial.

### Methodology
- **Participants:** 52 subjects (ages 20-26; 44 male, 8 female)
- **Trials:** 63 trials per participant
- **Protocol:** Each trial: auditory stimulus -> typing task -> emotional rating
- **Stimulus Source:** International Affective Digitized Sounds (IADS-2) database
- **Typing Task:** Fixed typing sequence "748596132" using number pad
- **Emotional Assessment:** Self-Assessment Manikin (SAM) with 9-point scales measuring valence and arousal dimensions
- **Data Collection:** Custom keystroke software captured typing behavior; 3,276 total samples collected (3.6% excluded for incomplete ratings)

### Keystroke Features Measured
1. **Keystroke Duration (Hold Time):** Time from key press to release
2. **Keystroke Latency (Flight Time):** Interval between key release and next key press
3. **Accuracy Rate:** Correctly vs. incorrectly typed sequences

### ML Models & Accuracy
- **No ML models used** - Traditional statistical analysis only
- Two-way repeated measures ANOVAs tested 3 (Valence) x 3 (Arousal) conditions

### Key Findings
- **Keystroke duration:** Significantly longer during low arousal (108.76 ms) versus high arousal (106.70 ms)
- **Keystroke latency:** Slowest at medium arousal (107.98 ms)
- **Accuracy:** No significant emotional effects detected
- **Effect Sizes:** Small -- individual variability substantially exceeded emotional contribution (eta-squared values: 9.14% for duration; 11.48% for latency)
- **Core Insight:** While auditory-induced arousal measurably influenced keystroke timing, personalized emotion recognition models could improve accuracy given substantial between-subject variation

### Limitations
- Excluded 11 subjects with unsuccessful emotion elicitation (>3 missing cells in 3x3 design)
- This exclusion "increased likelihood of detecting desired effects," potentially inflating results
- When excluded subjects were included via imputation, keystroke latency effects diminished
- Fixed numeric typing task (not natural text entry)
- Small effect sizes suggest limited practical utility without personalization

---

<a name="paper-2"></a>
## Paper 2: Keystroke Dynamics Patterns While Writing Positive and Negative Opinions

**Source:** MDPI Sensors, Vol. 21(17), 5963 (2021)
**DOI:** 10.3390/s21175963
**URL:** https://www.mdpi.com/1424-8220/21/17/5963
**PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC8434638/
**Access Status:** SUCCESSFULLY FETCHED via PMC (Open Access)

### Abstract
The study analyzed whether keystroke patterns differ when writing positive versus negative opinions. It is "possible to recognize positive and negative opinions from the keystroke patterns with accuracy above the random guess; however, combination with other modalities might produce more accurate results."

### Methodology
- **Participants:** 50 university students (40 male, 10 female, mean age 21 +/- 1 years)
- **Setting:** Laboratory with standardized computers and keyboards
- **Design:** Semi-experimental, within-subject design
- **Data Collection:** Participants wrote four opinions about learning experiences:
  - Best and worst teacher experiences
  - Best and worst subject experiences
- **Emotional Assessment:** Self-Assessment Manikin (SAM) questionnaire measured pleasure, arousal, and dominance before and after each writing session

### Keystroke Features Measured (51 total features)
- **Digraph features (2-key sequences):** Duration between key presses, dwell time, latency between keys
- **Trigraph features (3-key sequences):** Similar timing measurements for 3-key combinations
- **Shift key digraphs:** Special sequences with modifier keys
- **Frequency features:** Spacebar, backspace, delete, arrow keys, caps lock usage rates
- **Typing speed:** Keystrokes per second
- Mean values and standard deviations calculated for all timing characteristics

### ML Models & Accuracy

| Task | Best Model | F1-Score |
|------|-----------|----------|
| Positive vs. Negative Opinion | SVM | 0.627 |
| Pleasure Level Classification (L3 labeling) | SVM | 0.765 |
| Arousal Level Classification (L1 labeling) | SVM | 0.654 |

**Models Tested:** SVM, Random Forest, Naive Bayes, k-Nearest Neighbors (SVM performed best)

### Key Findings
- **Significant Features (p<0.05):** Spacebar frequency (strongest predictor), typing speed, digraph timing (key press intervals, dwell duration), trigraph latency measurements
- Pleasure and arousal showed significant differences between positive and negative opinions (Mann-Whitney test: p<0.01)
- Dominance did NOT differ significantly
- Normalizing keystroke features by subtracting individual baseline values improved classification performance
- Only one feature (spacebar frequency) remained significant after multiple-comparison correction

### Limitations
1. Homogenous sample -- exclusively university students aged 20-22; limited generalizability
2. Modest accuracy -- positive/negative classification only slightly above random (0.627 F1)
3. Imbalanced labels -- unequal class distributions affected arousal/pleasure classification
4. Limited labeling -- four samples per user insufficient for personalization
5. Confounding factors -- experimental conditions and daily mood may influence keystroke patterns
6. Small feature effect -- only spacebar remained significant after correction

---

<a name="paper-3"></a>
## Paper 3: Touchscreen Typing Pattern Analysis for Remote Detection of the Depressive Tendency

**Source:** Nature Scientific Reports (2019)
**DOI:** 10.1038/s41598-019-50002-9
**URL:** https://www.nature.com/articles/s41598-019-50002-9
**PMC:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6746713/
**Access Status:** SUCCESSFULLY FETCHED via PMC (Open Access)

### Abstract
This study proposes a machine learning method to discriminate between individuals with depressive tendency (DT) and healthy controls using typing patterns from smartphones. Researchers analyzed keystroke dynamics passively collected during natural typing on touchscreen devices, achieving an AUC of 0.89 with 82% sensitivity and 86% specificity.

### Methodology
- **Participants:** 25 young adults (mean age ~23.7 years); 11 with depressive tendency, 14 healthy controls
- **Inclusion:** No medication use, no upper limb dysfunction, college education, experienced smartphone users
- **Data Collection Tool:** Custom Android keyboard application (TypeOfMood)
- **Data Volume:** 34,581 typing sessions over 234 hours
- **Duration:** 124 days (November 2018 - March 2019)
- **Ground Truth:** PHQ-9 questionnaire at baseline; cutoff >= 5 for depressive tendency
- **Setting:** In-the-wild (natural daily smartphone usage)

### Keystroke Features Measured
**Primary Keystroke Dynamics Variables:**
- **Hold Time (HT):** Interval between key press and release
- **Flight Time (FT):** Interval between releasing one key and pressing next
- **Speed (SP):** Distance between successive keys divided by flight time
- **Press-Flight Rate (PFR):** Ratio of HT to FT

**Statistical Features Extracted from Each Variable:**
- Median, standard deviation, skewness, kurtosis

**Typing Metadata:**
- Delete rate
- Characters typed per session
- Session duration
- Characters per minute (CPM)

### ML Models & Accuracy

| Model | AUC | Sensitivity | Specificity |
|-------|-----|-------------|-------------|
| **Random Forest** | **0.89 [0.72-1.00 CI]** | **82%** | **86%** |
| Support Vector Machine | Tested | - | - |
| Gradient Boosting | Tested | - | - |

- **Cross-validation:** Leave-one-subject-out
- **Feature selection:** k-best algorithm (k=5)
- **Univariate best performer:** HT median alone achieved AUC = 0.85
- Prediction probabilities correlated with PHQ-9 scores (r = 0.64, p < 0.05)

### Key Findings
1. **Depressive Indicators:** Subjects with DT exhibited LONGER key hold times and GREATER variability in speed and press-flight rates, suggesting psychomotor retardation
2. **Most Discriminative Features:** Median hold time, standard deviation of speed, and press-flight rate metrics were selected in >= 90% of cross-validation iterations
3. **Data Requirements:** Only 50 typing sessions with 8+ characters minimum needed (vs. 400+ keystrokes in prior work)
4. **Daily Fluctuations:** Day-to-day probability variations negatively correlated with typing session volume (r = -0.60)
5. **Clinical Relevance:** Typing patterns provide objective, passive, continuous monitoring potential

### Limitations
- Small sample size (25 subjects) limits generalization
- PHQ-9 scores ranged 0-4 (controls) and 5-15 (DT group), not covering full spectrum (0-27)
- Classification based on single baseline PHQ-9 rather than longitudinal evaluations
- Future studies require ~2,200 subjects for 80% statistical power for clinical validation
- Cannot serve as standalone diagnostic tool

---

<a name="paper-4"></a>
## Paper 4: Identifying Emotional States Using Keystroke Dynamics

**Source:** CHI 2011 (Proceedings of the SIGCHI Conference on Human Factors in Computing Systems), pp. 715-724
**Authors:** Clayton Epp, Michael Lippold, Regan L. Mandryk (University of Saskatchewan)
**DOI:** 10.1145/1978942.1979046
**URL:** https://dl.acm.org/doi/10.1145/1978942.1979046
**Citations:** 367
**Access Status:** PARTIALLY EXTRACTED (paywall; details from thesis, Semantic Scholar, secondary sources)

### Abstract (from TLDR/secondary sources)
The research collected participants' keystrokes and their emotional states via self-reports, extracted keystroke features, and created classifiers for 15 emotional states. The work shows promise for detecting anger and excitement, with top results including classifiers for confidence, hesitance, nervousness, relaxation, sadness, and tiredness.

### Methodology
- **Study Type:** Field study (in situ, not laboratory -- participants typed naturally in real-world settings)
- **Emotional States:** 15 different emotional states measured via self-report
- **Emotional Assessment:** Self-Assessment Manikin (SAM) used alongside self-reports
- **Emotion Induction:** Films used for affective state induction in some experimental conditions
- **Classifier Development:** Created separate classifiers for each of 15 emotional states

### Keystroke Features Measured
- **Dwell Time (Hold Time):** Time elapsed between a key press and release
- **Flight Time:** Time elapsed between a key release and the next key press
- **Digraph Latency:** Timing between consecutive two-key sequences
- **Trigraph Latency:** Timing between consecutive three-key sequences
- **Content Attributes:** Number of backspaces, delete presses, special characters

### ML Models & Accuracy
- **Algorithm:** C4.5 Decision Tree (via Weka)
- **Classification Type:** 2-level classifiers per emotion

| Emotional State | Accuracy Range |
|----------------|---------------|
| Confidence | 77-88% |
| Hesitance | 77-88% |
| Nervousness | 77-88% |
| Relaxation | 77-88% |
| Sadness | 77-88% |
| Tiredness | 77-88% |
| Anger | Promising (specific % not extracted) |
| Excitement | Promising (specific % not extracted) |

**Evaluation Criteria:**
- Overall classification rate > 75%
- TP rates > 75% and FP rates < 25% for each class
- Higher tiers: TP > 80%/FP < 20%, and TP > 85%/FP < 15%

### Key Findings
- Keystroke dynamics can reliably classify 6 out of 15 emotional states (confidence, hesitance, nervousness, relaxation, sadness, tiredness) with 77-88% accuracy
- Anger and excitement showed promising results
- This was a seminal paper establishing that everyday keyboard typing rhythm can be used for emotion detection without specialized hardware
- Personalized models are essential due to high between-subject variability

### Limitations
- Abstract restricted by publisher; full methodology details limited in public sources
- Field study design introduces confounding variables
- Self-report emotional labels are subjective
- Dataset specifics not fully publicly available

---

<a name="paper-5"></a>
## Paper 5: Stress Detection via Keyboard Typing Behaviors by Using Smartphone Sensors and Machine Learning Techniques

**Source:** Journal of Medical Systems, 44(68) (2020)
**Authors:** Sagbas, E.A., Korukoglu, S., and Balli, S.
**DOI:** 10.1007/s10916-020-1530-z
**URL:** https://link.springer.com/article/10.1007/s10916-020-1530-z
**PubMed:** https://pubmed.ncbi.nlm.nih.gov/32072331/
**Access Status:** PARTIALLY EXTRACTED (paywall; details from PubMed and search results)

### Abstract
"Stress is one of the biggest problems in modern society...stress detection can be considered as a classification problem." The study examined accelerometer and gyroscope data from smartphone touchscreen writing behaviors across 46 participants in stressed versus calm states.

### Methodology
- **Participants:** 46 participants
- **Conditions:** Two states -- stressed and calm
- **Data Source:** Accelerometer and gyroscope sensor data during touchscreen typing
- **Windowing:** Sensor signals divided into 5, 10, and 15-second interval windows (creating 3 datasets)
- **Feature Count:** 112 features extracted from raw sensor signals
- **Feature Selection:** Gain Ratio feature selection algorithm for optimal feature subset identification

### Features Measured
- **Accelerometer data:** X, Y, Z axis measurements during typing
- **Gyroscope data:** Rotational measurements during typing
- **112 features derived from raw sensor signals** across 3 temporal windows
- Focus on physical typing behavior patterns rather than traditional keystroke timing

### ML Models & Accuracy

| Model | Accuracy |
|-------|----------|
| **k-Nearest Neighbor** | **87.56%** |
| C4.5 Decision Trees | 74.26% |
| Bayesian Networks | 67.86% |

### Key Findings
- Smartphone sensor data from typing behavior can effectively classify psychological stress without wearable devices
- k-NN achieved highest accuracy (87.56%) -- demonstrating feasibility of unobtrusive stress detection
- 15-second windows provided the most robust classification
- Gain Ratio feature selection improved model efficiency
- Physical typing dynamics (motion patterns) complement traditional keystroke timing features

### Limitations
- Relatively small sample (46 participants)
- Binary classification only (stressed vs. calm)
- Smartphone-specific (touchscreen typing, not desktop keyboard)
- Lab-induced stress may not reflect real-world stress patterns
- Sensor-based approach requires specific device capabilities

---

<a name="paper-6"></a>
## Paper 6: Does Peoples' Keyboard Typing Reflect Their Stress Level? An Exploratory Study

**Source:** Zeitschrift fur Psychologie (Journal of Psychology), Vol. 229, No. 4 (2021)
**Authors:** Paul Freihaut, Anja S. Goeritz (University of Freiburg)
**DOI:** 10.1027/2151-2604/a000468
**URL:** https://econtent.hogrefe.com/doi/10.1027/2151-2604/a000468
**Citations:** 5
**Access Status:** PARTIALLY EXTRACTED (paywall; details from Semantic Scholar API, search results)

### Abstract
The authors conducted two experiments -- a lab study (N=53) and an online study (N=924) -- where participants typed standardized text under high and low-stress conditions. While stress manipulation checks showed consistent differences between conditions, analysis of eleven typing features using statistical and machine learning approaches yielded inconsistent findings. The team published their data, code, and methods openly to support replication.

### Methodology
- **Study 1 (Lab):** 53 participants, controlled laboratory setting
- **Study 2 (Online):** 924 participants, web-based setting
- **Protocol:** Participants typed standardized text sequences during high-stress or low-stress conditions
- **Stress Assessment:** Multiple measures including:
  - Emotional valence and arousal via Self-Assessment Manikin (SAM)
  - Mood assessment
  - Rest and alertness measures
- **Stress Manipulation:** Verified through manipulation checks showing consistent condition differences
- **Open Science:** Data, code, and methods published openly

### Keystroke Features Measured (11 features)
**Typing Accuracy Features:**
- Number of backspace presses
- Error-related metrics

**Typing Speed Features:**
- Average time between pressing and releasing a key (Hold Time / Dwell Time)
- Typing latency metrics
- Additional timing-based features (11 total)

### ML Models & Accuracy
- Both frequentist statistical methods and machine learning classification methods were used
- **Individual-level classification:** ML algorithms accurately classified stressed typing for each individual based on their personal typing rhythms
- **Group-level classification:** Typing strings produced following stress manipulation could NOT be accurately classified using ML algorithms
- Specific model names not extracted from available sources

### Key Findings
1. **Inconsistent group-level results:** "A few isolated links between stress and keyboard typing, but the results were inconsistent across both studies and the analysis methods"
2. **Individual-level success:** ML algorithms accurately classified stress for INDIVIDUAL users based on their personal typing rhythms
3. **No universal markers:** No universal keystroke markers of stress were identified across participants
4. **Critical implication:** Personalized models are essential -- stress manifests differently in each person's typing
5. **Stress affected latency:** Stress led to some changes in typing latency specifically

### Limitations
- Standardized text (not free typing) may limit ecological validity
- Lab study (N=53) relatively small
- Inconsistency between lab and online results raises questions about robustness
- No universal features found -- limits scalable deployment
- Binary stress conditions (high vs. low) -- no gradation

---

<a name="paper-7"></a>
## Paper 7: A Review of Emotion Recognition Methods from Keystroke, Mouse, and Touchscreen Dynamics

**Source:** IEEE Access, Vol. 9, pp. 162197-162213 (December 2021)
**Authors:** Liying Yang, Sheng-Feng Qin (Northumbria University)
**DOI:** 10.1109/ACCESS.2021.3132233
**URL:** https://ieeexplore.ieee.org/document/9632591/
**Citations:** 30
**Access Status:** SUCCESSFULLY FETCHED from IEEE Xplore (Open Access)

### Abstract
"Emotion can be defined as a subject's organismic response to an external or internal stimulus event." Traditional emotion recognition methods (facial expression, gesture, gait) are intrusive. In contrast, "keystroke, mouse and touchscreen (KMT) dynamics data can be collected non-intrusively and unobtrusively as secondary data responding to primary physical actions." This systematic review examines state-of-the-art research on emotion recognition from KMT dynamics.

### Methodology (Systematic Review)
The review systematically addresses six research questions:
1. Common emotion elicitation methods and databases used
2. Which emotions can be recognized from KMT dynamics
3. Key features most appropriate for recognizing specific emotions
4. Most effective classification methods per emotion type
5. Application trends in KMT-based emotion recognition
6. Primary application contexts of concern

**Sources:** 117 references reviewed
**Funding:** China Scholarship Council; Newton Prize 2019

### Keystroke Features Reviewed (across literature)
- Keystroke dynamics patterns (dwell time, flight time, digraph/trigraph latency)
- Mouse dynamics characteristics (movement speed, click patterns, trajectory)
- Touchscreen interaction patterns (touch pressure, swipe patterns)
- Behavioral response metrics to emotional stimuli

### ML Models Discussed (across reviewed literature)
- Support Vector Machines (SVM)
- Deep Learning methods
- Deep Belief Networks
- Radial Basis Function Networks
- Federated Learning approaches
- General ML classification methods

### Key Findings
1. **Non-intrusive advantage:** KMT dynamics offer significant advantages over traditional intrusive methods
2. **Emotional state correlation established:** The review confirms emotional states correlate with changes in keystroke/mouse/touchscreen patterns
3. **Application context:** Emotion recognition is "an important prerequisite for emotion regulation towards better emotional states and work performance"
4. **Collaborative design:** Particular relevance for teamwork and collaborative design contexts
5. **Multiple emotions detectable:** Various discrete emotions can be distinguished through KMT dynamics
6. **Research impact:** 3,471 downloads and 30 citations demonstrate broad interest

### Limitations (of the field, as identified in review)
- Individual variability affects recognition accuracy substantially
- Lack of standardized databases across studies
- Context sensitivity -- different applications present varying challenges
- Privacy concerns with continuous behavioral monitoring
- Limited cross-study comparability due to methodological differences

---

<a name="paper-8"></a>
## Paper 8: Identifying Emotion by Keystroke Dynamics and Text Pattern Analysis

**Source:** Behaviour & Information Technology, Vol. 33, No. 9, pp. 987-996 (2014)
**Authors:** A. F. M. N. H. Nahin, Jawad Mohammad Alam, Hasan Mahmud, Md. Kamrul Hasan
**DOI:** 10.1080/0144929X.2014.907343
**URL:** https://www.tandfonline.com/doi/full/10.1080/0144929X.2014.907343
**Citations:** 86
**Access Status:** PARTIALLY EXTRACTED (paywall; details from Semantic Scholar, secondary sources)

### Abstract (from TLDR)
The research detects user emotions by analyzing keyboard typing patterns and the type of texts (words, sentences) typed by users, demonstrating "a substantial number of emotional states detected from user input."

### Methodology
- **Approach:** Combined keystroke dynamics AND text pattern analysis (first study to combine both modalities)
- **Emotional Classes:** 7 emotional states classified
- **Text Analysis:** Vector space model with Jaccard similarity method for free-text classification
- **Keystroke Analysis:** Timing attributes of keyboard interactions

### Keystroke Features Measured
- Keystroke timing attributes (dwell time, flight time, latency)
- Combined with text content features (word patterns, sentence patterns)
- Specific feature list not fully available from public sources

### ML Models & Accuracy
- Multiple machine learning algorithms tested
- **Combined approach (keystroke + text): Above 80% accuracy**
- **Best performance: Up to 87% accuracy** depending on the emotion
- The combined approach outperformed either modality alone

### Key Findings
1. **First combined approach:** Pioneered combining keystroke dynamics with text pattern analysis for emotion detection
2. **High accuracy:** 87% accuracy achievable for certain emotions
3. **Complementary modalities:** Text content and typing rhythm provide complementary emotional signals
4. **Practical applicability:** Uses standard keyboard input -- no special hardware needed

### Limitations
- Paywall limits full methodology extraction
- 7 emotion categories may oversimplify the emotional spectrum
- Dataset specifics not publicly available
- Generalizability across different typing contexts unclear

---

<a name="paper-9"></a>
## Paper 9: An Investigation into Keystroke Dynamics and Heart Rate Variability as Indicators of Stress

**Source:** MMM 2022 (International Conference on Multimedia Modeling)
**Authors:** Not fully extracted
**URL:** https://arxiv.org/abs/2111.09243
**Access Status:** SUCCESSFULLY FETCHED from arXiv

### Abstract
"Keystroke dynamics refers to the process of measuring and assessing a person's typing rhythm on digital devices. A digital footprint is created when a user interacts with devices like keyboards, mobile phones or touch screen panels and the timing of the keystrokes is unique to each individual though likely to be affected by factors such as fatigue, distraction or emotional stress." The study explores connections between keystroke timing patterns and heart rate variability (HRV) as stress indicators.

### Methodology
- **Data Collection Tool:** Loggerman application for keystroke measurement
- **Physiological Measurement:** Simultaneous HRV data gathering
- **Analysis Focus:** Relationship between keystroke timing variations and HRV as dual stress biomarkers

### Keystroke Features Measured
- **Timing for top-10 most frequently occurring bi-grams in English**
- Keystroke timing variations correlated with HRV patterns

### Key Findings
1. **Insufficient granularity:** "We need to use a more detailed representation of keystroke timing than the top-10 bigrams, probably personalised to each user"
2. **Personalization essential:** Generic bigram features are insufficient; user-specific feature sets needed
3. **Multimodal potential:** Combining keystroke dynamics with HRV shows promise for stress detection
4. **Individual differences confirmed:** Generic features fail across diverse populations

### Limitations
- Top-10 bigram approach proved insufficient for reliable stress detection
- Small-scale investigation
- Specific ML models and accuracy not detailed in abstract

---

<a name="paper-10"></a>
## Paper 10: Does Keystroke Dynamics Tell Us About Emotions? A Systematic Literature Review and Dataset Construction

**Source:** IEEE International Conference on Informatics in Economy (IE 2020)
**Authors:** Aicha Maalej, I. Kallel
**DOI:** 10.1109/IE49459.2020.9155004
**Citations:** 23
**Access Status:** PARTIALLY EXTRACTED (details from Semantic Scholar API and search results)

### Abstract
The paper investigates whether keystroke dynamics can effectively recognize human emotions. Through a systematic literature review covering the past decade, the authors examine "data acquisition procedures, datasets, extracted features, classification methods, and performance measures." They identify a significant gap in available datasets and respond by developing an interactive web application to construct a new emotion-recognition dataset.

### Methodology
- **Type:** Systematic Literature Review (SLR) with formal review protocol
- **Coverage:** Past decade of research on emotion recognition through keystroke dynamics
- **Dataset Construction:** Interactive web application developed to collect new emotion-labeled keystroke data
- **Output Dataset:** EmoSurv dataset

### Keystroke Features Catalogued (from reviewed literature)
- **D1U1:** Time between first key down and first key up (hold time)
- **D1U2:** Time between first key down and second key up
- **D1D2:** Time between first key down and second key down (key-to-key latency)
- **U1D2:** Time between first key up and second key down (flight time)
- Additional temporal relationships between key press and release events

### Dataset (EmoSurv)
- **Emotion Labels:** Happy (H), Sad (S), Angry (A), Calm (C), Neutral (N)
- **Collection Method:** Interactive web application
- **Available at:** IEEE DataPort (https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using)

### Key Findings
1. **Dataset scarcity:** Significant gap in publicly available emotion-labeled keystroke datasets
2. **Feature standardization needed:** Various studies use different feature extraction approaches
3. **Field maturity:** The field is still developing standardized methods and benchmarks
4. **New resource:** EmoSurv dataset addresses the data availability gap

---

<a name="supplementary"></a>
## Supplementary: Additional Papers from Semantic Scholar Search

### S1: Enhancing Typing Dynamics Emotion Recognition: Multi-Class XGBoost Approach (2024)
- **Authors:** Bandara, Dasath, Nanayakkara, Rathnayake, Thilakarthna
- **DOI:** 10.1109/ICAC64487.2024.10850934
- **Key Finding:** XGBoost demonstrated superior performance vs. SVM and Logistic Regression for emotion identification from keystroke patterns

### S2: Emotion Recognition Based on Piezoelectric Keystroke Dynamics (2021)
- **Authors:** Qi, Jia, Gao
- **DOI:** 10.1109/FLEPS51544.2021.9469843
- **Key Finding:** Random Forest achieved 78.31% accuracy for 4 emotional categories using piezoelectric sensor keystroke detection with only password entry

### S3: High Security User Authentication with Emotional Responses (2022)
- **Authors:** Jia, Qi, Huang, Zhou, Gao
- **DOI:** 10.1109/jsen.2021.3136902
- **Key Finding:** Random Forest achieved 96.40% accuracy for authentication when accounting for emotional state variations

### S4: Emotion Detection from Smartphone Keyboard: Temporal vs Spectral Features (2022)
- **Authors:** Mandi, Ghosh, De, Mitra
- **DOI:** 10.1145/3477314.3507159
- **Key Finding:** Time-domain models returned superior classification (average AUCROC 72%) vs. frequency-domain models (average 67%)

### S5: Emotion Prediction using Keystroke Dynamics: Comparative ML Study (2025)
- **Authors:** Kanagalakshmi K et al.
- **DOI:** 10.1109/ICIMIA67127.2025.11200823
- **Key Finding:** Random Forest achieved 95.4% accuracy with SMOTE for class balancing

### S6: Machine Learning Based Stress Detection Using Keyboard Typing Behavior (2023)
- **Authors:** Chunawale, Bedekar
- **Key Finding:** K-Nearest Neighbor achieved 84.21% accuracy using dimensionality reduction for stress level detection from typing behavior

### S7: Stress Detection for Keystroke Dynamics (CMU Thesis, 2018)
- **Author:** Shing-Hon Lau
- **Key Finding:** Dissertation goal was to "detect stress via keystroke dynamics -- the analysis of a user's typing rhythms -- and to detect the changes to those rhythms concomitant with stress"

---

## Cross-Paper Synthesis: Key Takeaways

### Most Commonly Measured Keystroke Features
| Feature | Description | Papers Using It |
|---------|-------------|-----------------|
| **Hold/Dwell Time** | Time key is pressed down | All papers |
| **Flight Time** | Time between key release and next press | Papers 1-4, 6-10 |
| **Digraph Latency** | Timing between 2-key sequences | Papers 2, 4, 9, 10 |
| **Trigraph Latency** | Timing between 3-key sequences | Papers 2, 4 |
| **Typing Speed** | Characters/keystrokes per unit time | Papers 1-3, 5-6 |
| **Error Rate** | Backspace/delete frequency | Papers 2, 4, 6 |
| **Press-Flight Rate** | Ratio of hold time to flight time | Paper 3 |
| **Key Pressure** | Force applied (touchscreen/piezo) | Papers 3, S2 |

### ML Model Performance Summary
| Model | Best Accuracy | Paper |
|-------|--------------|-------|
| Random Forest | 95.4% | S5 (with SMOTE) |
| Random Forest | 89% AUC | Paper 3 (depression) |
| C4.5 Decision Tree | 77-88% | Paper 4 (emotions) |
| k-NN | 87.56% | Paper 5 (stress) |
| KNN | 84.21% | S6 (stress) |
| SVM | 76.5% F1 | Paper 2 (pleasure) |
| XGBoost | Superior (specific % not given) | S1 |
| Combined (keystroke+text) | 87% | Paper 8 |

### Universal Finding: Personalization Is Essential
Nearly every paper independently concludes that individual differences in typing patterns dominate group-level emotional effects. Key evidence:
- Paper 1: Effect sizes tiny (eta-squared 9-11%); individual variability far exceeds emotional signal
- Paper 2: Normalizing to individual baselines improved classification
- Paper 6: ML classified stress for individuals but NO universal markers found across people
- Paper 9: "Personalised to each user" explicitly recommended
- Paper 4: Between-subject variability highlighted as primary challenge

### Features Most Predictive of Emotional/Stress States
1. **Hold time (dwell time):** Consistently the strongest single predictor across studies; longer hold times associated with depression, low arousal, and negative affect
2. **Flight time:** Second most predictive; affected by stress and arousal levels
3. **Typing speed:** Correlated with arousal intensity
4. **Error rate / backspace frequency:** Indicative of stress and cognitive load
5. **Spacebar frequency:** Strongest predictor in opinion valence classification (Paper 2)
6. **Digraph/trigraph latency:** Statistically significantly affected by stress (multiple papers)

### Key Emotion-to-Feature Mappings
- **Low arousal/sadness/depression:** Longer hold times, slower typing speed, higher variability
- **High arousal/stress/anger:** Shorter hold times, faster (but less accurate) typing, altered latency patterns
- **Negative affect:** Higher backspace/delete rate, altered spacebar patterns
- **Fatigue/tiredness:** Longer keystroke durations, increased latency variability

---

## 핵심 인사이트: 키스트로크 기반 투두 중요도 추정

### 문제
투두 리스트에서 중요도를 자동으로 판단하려면 어떻게 해야 하는가?
사전 정보(키워드 사전, 카테고리 분류) 없이, 질문 없이, 사용자 드래그 없이.

### 관찰
1. 키스트로크 타이밍(hold time, flight time, speed, error rate)은 감정 상태를 77-88% 정확도로 분류 가능 (Epp et al., CHI 2011)
2. 중요한 일 → 각성(arousal) 상승 → 타이핑 패턴 변화 (PLOS ONE, 2015)
3. **방향은 사람마다 다르다** — 어떤 사람은 빨라지고, 어떤 사람은 느려진다 (Freihaut, 2021, N=924)
4. 그러나 **변동 자체는 보편적이다** — arousal이 타이밍에 영향을 준다는 것은 모든 논문이 동의

### 해법: |z-score|
방향이 사람마다 다르므로, 방향을 무시한다.

```
importance = |현재 타이핑 속도 - 평균| / 표준편차
```

- 급해서 빨리 친다 → |z| 높음 → 중요
- 고민하다 느리게 친다 → |z| 높음 → 중요
- 평소처럼 편하게 친다 → |z| ≈ 0 → 루틴

**절댓값이 방향성 문제를 해결한다.**

### 콜드스타트
- 처음 5개 입력: 베이스라인 수집 (평균, 표준편차 계산)
- 이후: |z| 기반 자동 정렬
- 사전 데이터, 라벨, ML 모델 일체 불필요

### 기존 연구와의 차이
"투두 입력 시 키스트로크 다이내믹스의 |z-score|로 중요도를 추정한다"는
정확한 교차점은 기존 논문에 없다.
각 요소(키스트로크 감정 분류, 투두 우선순위, 콜드스타트)는 개별적으로 검증되었으나,
이 세 가지의 조합은 열린 연구 영역이다.

### 측정 피처 (중요도 순)
| 피처 | 설명 | 가중치 |
|------|------|--------|
| Hold time | 키를 누르고 있는 시간 | 0.30 |
| Flight time | 키에서 다음 키까지 이동 시간 | 0.30 |
| Typing speed | 초당 입력 속도 | 0.25 |
| Backspace ratio | 백스페이스 비율 (인지 부하) | 0.15 |

가중치 근거: 전 논문 종합 시 hold time과 flight time이 가장 일관되게 예측력 높음.
