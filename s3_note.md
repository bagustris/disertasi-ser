### S3 Note
day.month.year

---
to be answered:
- what is semantic primitive?
- what is prosodic feature?
- what is lexicon?
- spectral feature: feautes based on/extracted from spectrum
- normalization: normalize the waveform (divided by biggest amplitude)
- what is para and non-linguistic
- SVM classifier (vs Fuzzy??)
- idea: use DNN and DNN+Fuzzy for classification
- resume: all method need to be confirmed with other datasets

---
to study:
- statistical significance test
- idea: record emotional utterence freely from various speaker, find the similar words
- reverse the idea above: provided utterence, spoke with different emotion

---
todo:

- Blog about pengenalan emosi (read related reference)
- Investigate tdnn in iban
---

### Semantik
se.man.tik /sèmantik/
n Ling ilmu tentang makna kata dan kalimat; pengetahuan mengenai seluk-beluk dan pergeseran arti kata
n Ling bagian struktur bahasa yang berhubungan dengan makna ungkapan atau struktur makna suatu wicara

From wikipedia:
Semantic primes or semantic primitives are semantic concepts that are innately understood, but cannot be expressed in simpler terms. They represent words or phrases that are learned through practice, but cannot be defined concretely. For example, although the meaning of "touching" is readily understood, a dictionary might define "touch" as "to make contact" and "contact" as "touching", providing no information if neither of these words are understood.

alternative:

- **Multi-language emotion recognition based on acoustic and non-acoustic feature**
- A study to construct affective speech translation
 
Fix: **Speech emotion recognition from acoustic and contextual feature**

---
to study: correlation study of emotion dimension from acoustic and text feature

---
7.11.2017

- It is almost impossible to develop speech recognition using matlab/gnu octave due to data size and computational load
- Alternatives: KALDI and tensorflow, study and blog about it Gus!

---
10.11.2017

prosody: the patterns of stress and intonation in a language.

supresegmental: denoting a feature of an utterance other than the consonantal and vocalic components, for example (in English) stress and intonation.

Segment: is "any discrete unit that can be identified, either physically or auditorily".

low-rank matrix: approximation is a minimization problem, in which the cost function measures the fit between a given matrix (the data) and an approximating matrix (the optimization variable), subject to a constraint that the approximating matrix has reduced rank.--> represent music

sparse matrix or sparse array is a matrix in which most of the elements are zero. By contrast, if most of the elements are nonzero, then the matrix is considered dense. --> represent

---
24.11.2017

Pre-processing >> remove low part energy

---
12.04.2017

text processing:
- input: sentence (from deep learning)
- output: total VAD in sentence from each word

---
26.04.2017 
Philosophy of Doctoral study: Acoustic and Text feature for SER
1. Human recognize emotion from not only, but also word
2. Text feature can be extracted from speech by using Speech Recognition/STT
3. Having more information tends to improve SER performance
