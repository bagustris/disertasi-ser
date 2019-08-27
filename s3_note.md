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
2019-08-27:  
- Working on APSIPA draft, re running the experiment, found that longer window size (200ms) yield better result. Why? I don't know (need to be asked to Sensei?). But it is interesting to investigate: the impact of window size on feature extraction for speech emotion recognition.
- TODO: extract feature using 200ms window on IEMOCAP mono speech file (currently from stereo)


2019-08-26:  
- Minor reserch report accepted by advisor
- Forget lesson learned from ASJ autumn compautation: stack 2 RNNs with return sequences true, no dense layer after it, instead, use Flatten.
---

2019-08-16:  
- writing minor research report 3 (ANEW, Sentiwordnet, VADER)
- things to do: implement median and Mika Method (Mining valence, arousal....) for ANEW and Sentiwordnet.

2019-08-09:  
- Linear vs Logistic regression: The outcome (dependent variable) has only a limited number of possible values. Logistic regression is used when the response variable is categorical in nature.
- Accomplishment: Affect-based text emotion recognition using ANEW, VADER and Sentitowordnet. Current results shows VADER give best in term of CCC (for Valence). It is interesting that text give better score on valence while speech resulting worst score on valence compared to (CCC) score on arousal and dominance.

2019-08-06:  
- weekly meeting report: as presented (see in progress dir)
- weekly meeting note: what do you want to write for journal?
  - dimensional SER (speech + text) using recursive SVR
  - SER based on selected region (using attentional?)
- today's accomplishment: MAE and MSE from emobank using affective dictionary (ANEW).
- Q: What's different between lemmatization and stemming?
- A: Stemming and Lemmatization both generate the root form of the inflected words. The difference is that stem might not be an actual word whereas, lemma is an actual language word. Example: (connected, connection) --> connect, (am,is, are) --> be.

---
2019-07-12:  
- Writing draft for ASJ Autumn 2019
- Dimensional speech emotion recognition works with comparable CCC score [0.306, 0.4, 0.11].
- If you have been asked with simple question, answer with simple answer.
- Change presentation style on lab meeting: purpose it to investigate, analyze/discuss result. Not 
just to show.

---
2019-07-02:  
- working on multi-loss, multi-task learning, and multi modal fusion emotion recognition for AVEC2019.
- Flow: multi-loss+multi-task --> unimodal (early) --> bimodal (early) --> late --> combination of both.

---
2019-07-01:  
- dimensional ser on iemocap using Keras functional api works, it shows fairl good on MSE, but not on CCC.
- The current ccc for valence, arousal and dominance on test data is 0.03, 0.1, 0.05

---
2019-06-21:  
- philosophy: the ability to interrupt argument.

---
2019-06-19:  
- Feature-Level (FL) fusion (also known as “early fusion”) combines features from different modalities before performing recognition.
- Decision-Level (DL) fusion (also known as “late fusion”) combines the predictions and their probabilities given by each unimodal model for the multimodal model to make the final decision.

---  
2019-06-18:  
- Overall accuracy – where each sentence across the dataset has an equal weight, AKA weighted accuracy. In implementation this is the default accuracy in Keras metrics.
- Class accuracy – the accuracy is first evaluated for each emotion and then averaged, AKA unweighted accuracy. In implementaion, this class accuracy can be obtained by plotting normalized confustion matrix and get the average value along diagonal line.

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
- prosody (suprasegmental phonology): the patterns of stress and intonation in a language.   
- supresegmental: denoting a feature of an utterance other than the consonantal and vocalic components, for example (in English) stress and intonation.  
- Segment: is "any discrete unit that can be identified, either physically or auditorily".  
- low-rank matrix (e.g. rank-1: only one row independent): approximation is a minimization problem, in which the cost function measures the fit between a given matrix (the data) and an approximating matrix (the optimization variable), subject to a constraint that the approximating matrix has reduced rank.--> represent music  
- sparse matrix or sparse array is a matrix in which most of the elements are zero. By contrast, if most of the elements are nonzero, then the matrix is considered dense. --> represent what? speech? Yes, speech. (2019/07/30).

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

---  
2019-04-17
- Computation crash on yesterday run using egemaps feature, need to reduce size of feature
- run trimmed data with start-end silence removal, got lower accuracry (???)
- Pickle all data in iemocap dataset except for speech
- New dataset: [meld](https://github.com/SenticNet/MELD) ??
- __CONCEPT__: (acoustic) features are extracted from speech, i.e. wav file when offline, it
 is not make sense to extract feature from .npy or .pickle, that is just for simplification method. But, if we can avoid it (converting wav to pickle/npy for to save feature), do it. Pickle and npy still haold big memory (MB/GB).
 
 ---  
 2019-04-18  
 - Getting improvement of accuracy from baseline IEMOCAP with 5531 utterances without start-end trim by adding more features (40 and 44), i.e pitch(1) and formants (5). Reduce number of neuron on BLSTMwA (Bidirectional LSTM with Attention) system.
 - Doing start-end silence removal with `[10, 20, 30, 40, 50]` dB. ~~For 10 dB, need to change window size (due to shorten length of signal), compensate it with extending max length of feature sequence to 150 (original: 100).~~
 - Finding that running on GPU for this sequence data **SLOWER** than in CPU.
 - Add dropout 0.2 and 0.5 to the system, get higher accuracy. One simple way to detect overfitting is by checking val_loss vs loss, if it's higher, then overvitting (should be close each other). The cause usually is the number of trainable parameters is exceedingly greater than number of samples.
 - Found a paper about "tensor fusion", a method to unite multimodal data. Read it!

---  
2019-04-19  
- Found the better number of features: 40 (+1 F0, +5 Formants)
- With dropout of 0.5, feature with 50 dB start-end silence removal perform better (55%)

---  
2019-04-22
- Start-end silence removal can't give significant improvement on SER accuracy, move to add more features.
- replace LSTM with CuDNNLSTM to take advante of using GPU
- Use early stopping before model.fit to shortent computation
- Now evaluating on 39, 40 and 44 features
- __**concept**__: Overfitting occurs when number of trainable parameters greatly larger than number of samples, it is indicated with score of validation losss much higher than train loss.
- When to to stop iteration/epoch? when validation loss didn't decrease any more.

---  
2019-04-23  
- Need to model new feature that capture dynamics of speech sound if we want to improve SER performance
- Features to be tried: covarep, speechpy, with delta and delta-delta.

---  
2019-04-26  
- Building model for dimensional text emotion recognition. Currently using one output only (valence) and the obtained performance is still low. In term of MSE (mean squared error), the lowest mse was 0.522

---
2019-05-01  
- Multiple output VAD prediction workd on iemocap text, change the metric to mape (mean absolute percentage error), the lowest score is about 19%.
- Current result shows float number of VAD dimension, **need** to be cut only for .0 or .5. <-- no need

--- 
2019-05-09  
Today's meeting with Shirai-sensei:  
- Use input from affective dictionary for LSTM
- Concatenate output from sentiment with current word vector
- Try different affective dictionaries

---  
2019-05-16  
- Regression must use `linear` activation function
- Dimensonal SER works wit all 10039 utterances data, current best mape: 21.86%
- Prepare (presentation training) for lab meeting tomorrow
- The output of an LSTM is:
  - (Batch size, units) - with return_sequences=False
  - (Batch size, time steps, units) - with return_sequences=True

---  
2019-05-17  
- in math, logit function is simply the logarithm of the odds: logit(x) = log(x / (1 – x)).
- in tensorflow, logits is a name that it is thought to imply that this Tensor is the quantity that is being mapped to probabilities by the Softmax (input to softmax).
- end-to-end loss, minimize D1 (intra-personal) and maximize D2 (inter-personal), D1 and D2 is distant between (audio) embedding (in spekaker verification, need to be confirmed)
- Most MFCC uses 30 ms of window, this result spectral shape will the same for smaller. This is maybe why removing silence gives better performance.
- To capture the dynamics of emotion, maybe the use of delta and delta-delta will be better.
- Why removing will improve SER performance? Intuition. Silence is small noise, it may come from hardware, electrical of ambient noise. If it is included in speech emotion processing, the extracted feature may be not relevant because it extracts feature from small noise, not the speech. By removing this part, the extracted feature will only comes from speech not silence. Therefore, this is why the result better.

---
2019-05-19  
- **GRU** perform better and faster than LSTM. 
- Hence, CNN vs RNN --> RNN, LSTM vs GRU --> GRU. Global attention vs local attention --> ...?
- idea: Obtain local attention from waveform directly, only extract feature on two or more highest attentions.
- what's different between written text and spoken language (speech transcription)...?
- **Modern SNS and chat like twitter and facebook status is more similar to spoken language (as the concept of "twit") rather than written text, so it will be useful to analyze speech transcription than (formal) writtent text to analyse affect within that context.**

---
2019-05-25  
- evaluate word embedding method on iemocap text emotion recognition (word2vec, glove, fasstext), so far glove gives the best.
- In phonetics, rhythm is the sense of movement in speech, marked by the stress, timing, and quantity of syllables.   

---
2019-06-03  
- Progress research delivered (text emotion recognition, categorical & dimensional, written & spoken text)
- Text emotion recognition works well on dimensional, it is interpretable and easiler to be understood. Continue works on it.
- Combine acoustic and text feature for dimensional emotion recognition

---  
2019-06-04  
- re-run experiment on voice speech emotion recognition (ICSigsys 2019) for 0.1 threshold (using updated audiosegment)
- idea: how human brain process multimodal signal, implement it on computation

---  
2019-06-05  
RNN best practice:  
- Most important parameters: units and n layers
- Units (size): depend on data:
  - text data < 1 Mb --> < 300 units
  - text data 2 - 6 Mb --> 300-600 units
  - text data > 7 Mb --> > 700 units
- Units: 2 or 3 (source: Karpathy)
- Monitoring loss:  
  - Overfitting if: training loss << validation loss  
  - Underfitting if: training loss >> validation loss
  - Just right if training loss ~ validation loss
Problem with categorical emotion:
- Need balanced data
- To make balanced data, some context between utterances will gone/disappear

---
2019-06-04:  
- Idea: auditory based attention model for fusion of acoustic and text feature for speech emotion recognition. Attention is the main mechanism how to human auditory system perceive sound. By attention mechanism, human focus on what he interest to listen and collect the information from the sound, including emotion. In case speech emotion, human might focus on both tonal and verbal information. If the tonal information match the verbal information, than he believe the information he obtained is correct.
- To combine those information (verbal and tonal), two networks can be trained on the same label, separately. The acoustic network is the main (master/primary) and the text network is slave/secondary. The acoustic sytem acts as main system while the secondary system is supporting system which give weights to primary system. For categorical, If the weight above the thareshold (say 0.5 in the range 0-1), then both sytems agree for the same output/category. If no, the output of the system is the main system weighted by secondary system.
- For categorical (which is easier to devise), the output of the system is the main system weighted by secondary system (multiplication) ---> multiplicative attention?
- Whether it is additive or multiplication, beside via attention, it also can be implemented directly when combining two modalities. Instead of concatenate, we can use add() or multiply(). But, how to shape/reshape input feature?

---
2019-06-08:  
- As of 2016, a rough rule of thumb
is that a supervised deep learning algorithm will generally achieve acceptable
performance with around 5,000 labeled examples per category, and will match or
exceed human performance when trained with a dataset containing at least 10
million labeled examples. Working successfully with datasets smaller than this is
an important research area, focusing in particular on how we can take advantage
of large quantities of unlabeled examples, with unsupervised or semi-supervised
learning.

---  
2019-06-12:  
- Working on dimensional emotion recognition (for cocosda?), the result from acoustic and text feature only shows a little improvement compared to acoustic only for text only.
- Current architecture:
  - Acoustic: 2 stack BLSTM
  - Text: 2 stack LSTM
  - Combination: 2 Dense layers
- Current (best result): 
  - [mse: 0.4523394735235917, mape: 19.156075267531484, mae: 0.5276844193596124]
- Need advance strategy for combination: hfusion, attention, tensor fusion???
  
---
2019-06-14  
- Current result (train/val loss plot) shows that system is overfitting regardles complexity of architecture (even with smalles number of hyperparameter). Needs to re-design.
- As obtained previously, the more data the better data. How if the data is limited?
- If can't increase the number/size of data, maybe the solution is to increase the number of input features.
- Let's implement it, and see if it works.
- to do: implement CCC (concordance coeff.) on current sytem

---
2019-06-17:  
- Interspeech2019 --> rejected
- Usually people use 16-25ms for window size, especially when modeled with recursive structures.
- to study (a must): WA vs UA, weighted accuracy vs unweighted accuracy; WAR vs UAR (unweighted average recall)
- Accuracies by themselves are useless measure of goodness unless combined with precision values or averaged over Type I and II errors. What is those?
- Answer: see this: [https://en.wikipedia.org/wiki/Type_I_and_type_II_errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)
- it must never be concluded that, for example, 65.69% is "better" than 65.18%, or even that 68.83% is "better" that 63.86%, without providing a measure of significance for that statement


