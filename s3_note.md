### S3 Note: Journal of PhD Life
~~day.month.year~~  
year-month-day (ISO 8601)  
Content:  
- if possible, write per day  
- what is your idea, how to implement  
- what is the problem, what is your proposed solution
- what is X, what is Y
- what did you have done today, what is your plan for tomorrow

---
2017-10-09  
to be answered:
- what is semantic primitive?
- what is prosodic feature?
- what is lexicon?
- spectral feature: feautures based on/extracted from spectrum
- normalization: normalize the waveform (divided by biggest amplitude)
- what is para and non-linguistic
- SVM classifier (vs Fuzzy??)
- idea: use DNN and DNN+Fuzzy for classification
- resume: all method need to be confirmed with other datasets

---
2017-10-10  
to study:  
- statistical significance test
- idea: record emotional utterence freely from various speaker, find the similar words
- reverse the idea above: provided utterence, spoke with different emotion

---
todo:  
- Blog about pengenalan emosi (read related reference)
- Investigate tdnn in iban

---  
2017-10-11  
Semantik
se.man.tik /sèmantik/
n Ling ilmu tentang makna kata dan kalimat; pengetahuan mengenai seluk-beluk dan pergeseran arti kata
n Ling bagian struktur bahasa yang berhubungan dengan makna ungkapan atau struktur makna suatu wicara

From wikipedia:
Semantic primes or semantic primitives are semantic concepts that are innately understood, but cannot be expressed in simpler terms. They represent words or phrases that are learned through practice, but cannot be defined concretely. For example, although the meaning of "touching" is readily understood, a dictionary might define "touch" as "to make contact" and "contact" as "touching", providing no information if neither of these words are understood.

alternative:
- **Multi-language emotion recognition based on acoustic and non-acoustic feature**
- A study to construct affective speech translation

Fix: **Speech emotion recognition from acoustic and contextual feature**
to study: correlation study of emotion dimension from acoustic and text feature

---
2017-11-7  
- It is almost impossible to develop speech recognition using matlab/gnu octave due to data size and computational load
- Alternatives: KALDI and tensorflow, study and blog about it Gus!

---
2017-11-10  
prosody: the patterns of stress and intonation in a language.  
supresegmental: denoting a feature of an utterance other than the consonantal and vocalic components, for example (in English) stress and intonation.  
Segment: is "any discrete unit that can be identified, either physically or auditorily".  
low-rank matrix: approximation is a minimization problem, in which the cost function measures the fit between a given matrix (the data) and an approximating matrix (the optimization variable), subject to a constraint that the approximating matrix has reduced rank.--> represent music  
sparse matrix or sparse array is a matrix in which most of the elements are zero. By contrast, if most of the elements are nonzero, then the matrix is considered dense. --> represent

---
2017-11-24  
Pre-processing >> remove low part energy

---
2017-12-04  
text processing:
- input: sentence (from deep learning)
- output: total VAD in sentence from each word

---  
2018-04-26  
Philosophy of Doctoral study: Acoustic and Text feature for SER
1. Human recognize emotion from not only, but also word
2. Text feature can be extracted from speech by using Speech Recognition/STT
3. Having more information tends to improve SER performance

---  
2018-04-08  
Idea for thesis book:
1. Introduction
2. Speech emotion recognition: Dimensional Vs Categorical Approach
2. Deep learning based Speech emotion Recognition
3. Emotion recognition from Text
4. Combining acoustic and text feature 
5. Conclusion and future works

---  
2018-09-20  
Mid-term presentation:
1. What kind of direction this study will be proceeded in the future,
2. How important this study is in this direction, and
3. How much contributions can be expected

---  
2018-09-13  
Research idea to be conducted:
Are semantics contributes to perceived emotion recognition?
A listening test to test the hyphothesis
Listening test:
Speech only --> emotion recognition
Speech + transcription --> emotion recognition

---  
2018-10-11  
Course to be take in term 2-1:
1. Data Analytics
2. Analysis of information science

---  
2018-11-29  
Zemi:
Speker dependent vs speaker independent
Speaker depandet: The same speaker used for training and dev 
Speaker Independnet: The different speaker used for training and dev 

---  
2018-12-12  
a cepstral gain c0 is the logarithm of the modeling filter gain
loggging kaldi output:  
~/kaldi/egs/iban/s5 $ ./local/nnet3/run_tdnn.sh 2>&1 | tee run-tdnn.log
some solution of kaldi errors:
Error:
Iteration stops on run_tdnn.sh no memory
Solution:
You shouldn't really be running multiple jobs on a single GPU. 
If you want to run that script on a machine that has just one GPU, one 
way to do it is to set exclusive mode via 
sudo nvidia-smi -c 3 

and to the train.py script, change the option "--use-gpu=yes" to 
"--use-gpu=wait" 
which will cause it to run the GPU jobs sequentially, as each waits 
till it can get exclusive use of the GPU. 

Errror:  
Refusing to split data for number of speakers"
Solution:  
You didn't provide enough info, but in general, you cannot split the directory in more parts than the number of speakers is.
So if you called the decoding with -nj 30 and you have 25 speakers (you can count lines of the spk2utt file) this is the error you receive.

Show how many features extracted using mfcc:
~/kaldi-trunk/egs/start/s5/mfcc$ ../src/featbin/feat-to-dim ark:/home/k/kaldi-trunk/egs/start/s5/mfcc/raw_mfcc_train.1.ark ark,t:-

GMM (gaussian mixture model): A mixture of some gaussian distribution  

---  
2018-12-14  
Speech is not only HOW it is being said but also what is being said.
low-level feature (descriptor): extracted per frame. High level feature: extracted per utterance.
high-level feature: extracted per frame?

---
2018-12-17  
warning from python2:
/home/bagustris/.local/lib/python2.7/site-packages/scipy/signal/_arraytools.py:45: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  b = a[a_slice]

---
2018-12-18  
Idea: concurrent speech and emotion recognition  
Desc: Currently speech recognition and emotion recognition is two separated research areas. Researcher build and improve performance of speech recognition and emotion recognition independently such as works done by (\cite{}, \cite{}, and \cite{}). The idea is simple, both emotion and text (output of speech recognition) can be extracted from speech by using the same features. Given two labels, transcription and emotion, two tasks can be done simulatenously: speech recognition and emotion recognition by training acoustic features to map both text and emotion label.

idea for speech emotion recognition from acoustic and text features
1. train speech corpus with given transcription --> output: predicted VAD (3 values)
2. obatin VAD score from speech transcription --> output: predicted VAD (3 values)
3. Feed all 6 variables into DNN wih actual VAD value

---  
2018-12-20  
mora (モーラ): Unit in phonology that determine syllable weight  
Example: 日本、にほん、3 mora, but, にっぽん　is 3 mora  
morpheme: the smallest unit of meaning of a word that can be devided to:  
Example: like --> 1 morpheme, but unlikely is 3 morpheme (un, like, ly)  
Find the different between dynamic feature and static feature and its 
relation to human perception.  
How about statistic feature?  
notch noise = v-shaped noise...?  

---  
2018-12-27  
Loss function = objective functions  
How to define custom loss function?  
Here in Keras, https://github.com/keras-team/keras/issues/369  
But I think loss="mse" is OK  
note: in avec baseline, there is already ccc_loss  
dense and dropout layer:    
The dense layer is fully connected layer, so all the neurons in a layer are connected to those in a next layer. The dropout drops connections of neurons from the dense layer to prevent overfitting. A dropout layer is similar except that when the layer is used, the activations are set to zero for some random nodes  
povey window: povey is a window I made to be similar to Hamming but to go to zero at the edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85).

---
08.02.2019  
Likelihood vs probability:  
Likelihood is the probability that an event that has already occurred would yield a specific outcome. Probability refers to the occurrence of future events, while a likelihood refers to past events with known outcomes. Probability is used when describing a function of the outcome given a fixed parameter value.

---  
2019-02-15  
Idea: Provided dataset with speech and linguistic information, 
how human perceive emotion from emotional speech with and without linguistic information?

---  
2019-03-06  
Idea for ASJ autumn 2019: Emotional speech recognition
dataset: IEMOCAP
tools: DeepSpeech

---  
2019-04-04  
How to map emotion dimension to emotion category?
One solution is by inputting emotion dimension to machine learning tool, such as GMM.
Reda et al. tried this method and obtain very big improvement from 54% to 94% of accuracy.
Next, try deep learning methods.
Also, try to learn confusion matrix.

---  
17.02.2019 
Idea for ICSygSis 2019: Voiced Spectrogram and CNN for SER
idea: remove silence from speech.  
Finding:  Many pieces of data only contains noisy or silence, but labeled as neutoral or other emotion.
Next idea: add silence category as it is important cue for speech emotion recognition

---  
2019-04-08  
The research paper below shows the evidence that music didn't improve creativity.
https://onlinelibrary.wiley.com/doi/epdf/10.1002/acp.3532
How about if we change the experiment set-up. Listening music first, 5-10 minutes, 
stop, give the question. Intuition: While music didnot contribute to improve creativity, but it may contribute to mood and emotion. After being calm by listening, it may improve creativity.

---  
