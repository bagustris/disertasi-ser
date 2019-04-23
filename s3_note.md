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
- spectral feature: features based on/extracted from spectrum
- normalization: normalize the waveform (divided by biggest amplitude)
- what is para and non-linguistic
- SVM classifier (vs Fuzzy??)
- idea: use DNN and DNN+Fuzzy for classification
- resume: all method need to be confirmed with other datasets
- Entering JAIST as research student.

---
2017-10-10  
to study:  
- statistical significance test
- idea: record emotional utterence freely from various speaker, find the similar words
- reverse the idea above: provided utterence, spoke with different emotion

---
todo:  
- Blog about emotion recognition (indonesia:pengenalan emosi) by reading related reference.
- Investigate tdnn in iban

---  
2017-10-11  
Semantik
se.man.tik /sèmantik/ 
n Ling ilmu tentang makna kata dan kalimat; pengetahuan mengenai seluk-beluk dan pergeseran arti kata  
n Ling bagian struktur bahasa yang berhubungan dengan makna ungkapan atau struktur makna suatu wicara  

From wikipedia:
Semantic primes or semantic primitives are semantic concepts that are innately understood, but cannot be expressed in simpler terms. They represent words or phrases that are learned through practice, but cannot be defined concretely. For example, although the meaning of "touching" is readily understood, a dictionary might define "touch" as "to make contact" and "contact" as "touching", providing no information if neither of these words are understood.

alternative research theme:
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
- prosody: the patterns of stress and intonation in a language.   
- supresegmental: denoting a feature of an utterance other than the consonantal and vocalic components, for example (in English) stress and intonation.  
- Segment: is "any discrete unit that can be identified, either physically or auditorily".  
- low-rank matrix: approximation is a minimization problem, in which the cost function measures the fit between a given matrix (the data) and an approximating matrix (the optimization variable), subject to a constraint that the approximating matrix has reduced rank.--> represent music  
- sparse matrix or sparse array is a matrix in which most of the elements are zero. By contrast, if most of the elements are nonzero, then the matrix is considered dense. --> represent what? speech?

---
2017-11-24  
Pre-processing >> remove low part energy

---
2017-12-04  
text processing:
- input: sentence (from deep learning)
- output: total VAD in sentence from each word

---  
2018-04-08  
- Idea for thesis book:  
1. Introduction
2. Speech emotion recognition: Dimensional Vs Categorical Approach
2. Deep learning based Speech emotion Recognition
3. Emotion recognition from Text
4. Combining acoustic and text feature 
5. Conclusion and future works
- Starting PhD at JAIST, bismillah.

---  
2018-04-26  
Philosophy of Doctoral study: Acoustic and Text feature for SER
1. Human recognize emotion from not only, but also word
2. Text feature can be extracted from speech by using Speech Recognition/STT
3. Having more information tends to improve SER performance

---  
2018-09-13  
Research idea to be conducted:
- Are semantics contributes to perceived emotion recognition?
- A listening test to test the hyphothesis
Listening test:  
- Speech only --> emotion recognition
- Speech + transcription --> emotion recognition

---  
2018-09-20  
Mid-term presentation:
1. What kind of direction this study will be proceeded in the future,
2. How important this study is in this direction, and
3. How much contributions can be expected


---  
2018-10-11  
Course to be taken in term 2-1:
1. Data Analytics
2. Analysis of information science

---  
2018-11-29  
Zemi:   
- Speaker dependent vs speaker independent   
- Speaker dependent: The same speaker used for training and dev    
- Speaker Independent: The different speaker used for training and dev  

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
`sudo nvidia-smi -c 3`

and to the train.py script, change the option "--use-gpu=yes" to 
"--use-gpu=wait" 
which will cause it to run the GPU jobs sequentially, as each waits 
till it can get exclusive use of the GPU. 

Error:  
"Refusing to split data for number of speakers"
Solution:  
You didn't provide enough info, but in general, you cannot split the directory in more parts than the number of speakers is.
So if you called the decoding with -nj 30 and you have 25 speakers (you can count lines of the spk2utt file) this is the error you receive.  

Show how many features extracted using mfcc: 
~/kaldi-trunk/egs/start/s5/mfcc$ ../src/featbin/feat-to-dim ark:/home/k/kaldi-trunk/egs/start/s5/mfcc/raw_mfcc_train.1.ark ark,t:-

GMM (gaussian mixture model): A mixture of some gaussian distribution.  

---  
2018-12-14  
- Speech is not only HOW it is being said but also what is being said.  
- low-level feature (descriptor): extracted per frame.  
- High level feature: extracted per utterance.

---  
2018-12-17  
- warning from python2:  
/home/bagustris/.local/lib/python2.7/site-packages/scipy/signal/_arraytools.py:45: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  b = a[a_slice]

---
2018-12-18  
- Idea: concurrent speech and emotion recognition  
- Desc: Currently speech recognition and emotion recognition is two separated research areas. Researcher build and improve performance of speech recognition and emotion recognition independently such as works done by (\cite{}, \cite{}, and \cite{}). The idea is simple, both emotion and text (output of speech recognition) can be extracted from speech by using the same features. Given two labels, transcription and emotion, two tasks can be done simulatenously: speech recognition and emotion recognition by training acoustic features to map both text and emotion label.

Idea for speech emotion recognition from acoustic and text features:  
1. train speech corpus with given transcription --> output: predicted VAD (3 values)
2. obatin VAD score from speech transcription --> output: predicted VAD (3 values)
3. Feed all 6 variables into DNN with actual VAD value

---  
2018-12-20  
- mora (モーラ): Unit in phonology that determine syllable weight  
- Example: 日本、にほん、3 mora, but, にっぽん　is 4 mora  
- Morpheme: the smallest unit of meaning of a word that can be devided to (it is in linguistic, in acoustic the smallest unit is phoneme) .
- Example: like --> 1 morpheme, but unlikely is 3 morpheme (un, like, ly)    
- Find the different between dynamic feature and static feature and its 
- relation to human perception.  
- How about statistic feature?  
- notch noise = v-shaped noise...?  

---  
2018-12-27  
- Loss function = objective functions  
- How to define custom loss function?  
- Here in Keras, https://github.com/keras-team/keras/issues/369  
- But I think loss="mse" is OK  
- note: in avec baseline, there is already ccc_loss  
- Dense and dropout layer:    
The dense layer is fully connected layer, so all the neurons in a layer are connected to those in a next layer. The dropout drops connections of neurons from the dense layer to prevent overfitting. A dropout layer is similar except that when the layer is used, the activations are set to zero for some random nodes  
povey window: povey is a window I made to be similar to Hamming but to go to zero at the edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85).

---
2019-02-08    
- Likelihood vs probability:  
- Likelihood is the probability that an event that has already occurred would yield a specific outcome. Probability refers to the occurrence of future events, while a likelihood refers to past events with known outcomes. Probability is used when describing a function of the outcome given a fixed parameter value.

---  
2019-02-15  
- Idea: Provided dataset with speech and linguistic information, 
- How human perceive emotion from emotional speech with and without linguistic information?

---  
2019-02-17
- Idea for ICSygSis 2019: Voiced Spectrogram and CNN for SER
- idea: remove silence from speech.  
- Finding:  Many pieces of data only contains noisy or silence, but labeled as neutral or other emotion.
- Next idea: add silence category as it is important cue for speech emotion recognition (??)

---  
2019-03-06  
- Idea for ASJ autumn 2019: Emotional speech recognition  
- dataset: IEMOCAP  
- tools: DeepSpeech

---  
2019-04-04  
- How to map emotion dimension to emotion category?
- One solution is by inputting emotion dimension to machine learning tool, such as GMM.
- Reda et al. tried this method and obtain very big improvement from 54% to 94% of accuracy.
- Next, try deep learning methods.
- Also, try to learn confusion matrix.

---  
2019-04-08  
- The research paper below shows the evidence that music didn't improve creativity.  
https://onlinelibrary.wiley.com/doi/epdf/10.1002/acp.3532  
- How about if we change the experiment set-up. Listening music first, 5-10 minutes, stop, give the question.  
- Intuition: While music didnot contribute to improve creativity, but it may contributes to mood and emotion. After being calm by listening, it may improves creativity.

---  
2019-04-09  
Today, 
- I implemented F0 based voiced segmentation for feature extraction using YAPT method with `amfm_decompy` package (now running in my PC). 
- Learned how to convert data from tuple to 1D array (using np.flatten()), wrote blog post about it.
- Obtained signature from Akagi-sensei for MSP-Impro database, and forward it tu TU Dallas.
- Plan for tomorrow: run BSLTM from obtained feature today --> write result on WASPAA.

---  
2019-04-10  
- Attended workshop: deeplearning for supercomputer cray XC40 (xc40 /work/$USER/handson/)
- Run obtained feature (from F0) to currenty BLSTM+attention model system, got lower result. It may need to be processed per segment, not whole segment. Train each voiced segment feature, use majority voting like to decide.
- Prepare presentation for Lab meeting on Friday.
- Replace owncloud with nextcloud, now up to 300 GB.

---
2019-04-11  
- made slide for tomorrow lab meeting presentation.
- run obtained feature on BLSTM+attention model, the higher accuracy was 52%, still lower than previous.
- change window size from 20 ms to 0.1 s, 0.04, 0.08, etc. Find the best result.
- Email Prof. Busso, asking for the speech transcription.

---  
2019-04-12  
Today's lab meeting:
- Compared voiced and voiced+unvoiced part --> done?
- You study at the school of information science? what is science in your PhD?
  Human perceive emotion from speech. The speech constain some information, mainly : vocal tone information and lexical/linguistic information. Human can perceive emotion from speech only. In some cases it is difficult, like in noisy environment. Given another information, lexical information, it will be useful for human to recognize emotion of speaker. Can computer do that?
- Information science is a field primarily concerned with the analysis, collection, classification, manipulation, storage, retrieval, movement, dissemination, and protection of information.
- recognition of human emotion by computer is one area of information science, right Sensei?
- Text feature is feature from text data
  
---  
2019-04-13  
- In linguistics, prosody is concerned with those elements of speech that are not individual phonetic segments but are properties of syllables and larger units of speech. These are linguistic functions such as intonation, tone, stress, and rhythm.
- Extract F0 from IEMOCAP, padded with other 34 features, run it on PC, still got lower result.

--- 
2019-04-15  
- A voiced sound is category of consonant sounds made while the vocal cords vibrate. All vowels in English are voiced, to feel this voicing, touch your throat and say AAAAH. ... That is voicing. Consonants can be either [voice/unvoice](/fig/460.png)
- Perform start-end silence removal on 5331 IEMOCAP utterances

--- 
2019-04-16 
- Running experiment using feature from trimmed voice, still got lowe performance, 47%
- Extract egemamps feature set from IEMOCAP data, expecting improvement on SER system as egemaps is tailored for speech emotion recognition
- Running extracted egemaps feature on the system, 447,672,324 parameters, it breaks the GPU capability
- Next: extract egemaps feature from trimmed speeech: 10, 15, 20 dB  
- GPU error (out of mems): ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[872704,512] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc

2019-04-17
- Computation crash on yesterday run using egemaps feature, need to reduce size of feature
- run trimmed data with start-end silence removal, got lower accuracry (???)
- Pickle all data in iemocap dataset except for speech
- New dataset: [meld](https://github.com/SenticNet/MELD) ??
- __CONCEPT__: (acoustic) features are extracted from speech, i.e. wav file when offline, it
 is not make sense to extract feature from .npy or .pickle, that is just for simplification method. But, if we can avoid it (converting wav to pickle/npy for to save feature), do it. Pickle and npy still haold big memory (MB/GB).
 
 2019-04-18  
 - Getting improvement of accuracy from baseline IEMOCAP with 5531 utterances without start-end trim by adding more features (40 and 44), i.e pitch(1) and formants (5). Reduce number of neuron on BLSTMwA (Bidirectional LSTM with Attention) system.
 - Doing start-end silence removal with `[10, 20, 30, 40, 50]` dB. ~~For 10 dB, need to change window size (due to shorten length of signal), compensate it with extending max length of feature sequence to 150 (original: 100).~~
 - Finding that running on GPU for this sequence data **SLOWER** than in CPU.
 - Add dropout 0.2 and 0.5 to the system, get higher accuracy. One simple way to detect overfitting is by checking val_loss vs loss, if it's higher, then overvitting (should be close each other). The cause usually is the number of trainable parameters is exceedingly greater than number of samples.
 - Found a paper about "tensor fusion", a method to unite multimodal data. Read it!

2019-04-19  
- Found the better number of features: 40 (+1 F0, +5 Formants)
- With dropout of 0.5, feature with 50 dB start-end silence removal perform better (55%)

2019-04-22
- Start-end silence removal can't give significant improvement on SER accuracy, move to add more features.
- replace LSTM with CuDNNLSTM to take advante of using GPU
- Use early stopping before model.fit to shortent computation
- Now evaluating on 39, 40 and 44 features
- __**concept**__: Overfitting occurs when number of trainable parameters greatly larger than number of samples, it is indicated with score of validation losss much higher than train loss.
- When to to stop iteration/epoch? when validation loss didn't decrease any more.

2019-04-23  
- Need to model new feature that capture dynamics of speech sound if we want to improve SER performance
- Features to be tried: covarep, speechpy, with delta and delta-delta.

