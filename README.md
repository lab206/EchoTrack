### <p align="center"> EchoTrack: Auditory Referring Multi-Object <br /> Tracking for Autonomous Driving
<br>
<div align="center">
  Jiacheng&nbsp;Lin*</a> <b>&middot;</b>
  Jiajun&nbsp;Chen*</a> <b>&middot;</b>
  Kunyu&nbsp;Peng*</a> <b>&middot;</b>
  Xuan&nbsp;He</a> <b>&middot;</b>
  Zhiyong&nbsp;Li</a> &middot;</b>
  Rainer&nbsp;Stiefelhagen</a> &middot;</b>
  <a href="https://yangkailun.com/" target="_blank">Kailun&nbsp;Yang</a>
  
  <br> <br>
  <a href="https://arxiv.org/pdf/2402.18302.pdf" target="_blank">Paper</a>
</div>

<br>
<p align="center">Code will be released soon. </p>
<br>

<div align=center><img src="imgs/network.png" /></div>

### Update
- 2024.02.29 Init repository.

### Abstract
This paper introduces the task of Auditory Referring
Multi-Object Tracking (AR-MOT), which dynamically tracks
specific objects in a video sequence based on audio expressions and appears as a challenging problem in autonomous
driving. Due to the lack of semantic modeling capacity in
audio and video, existing works have mainly focused on text-
based multi-object tracking, which often comes at the cost of
tracking quality, interaction efficiency, and even the safety of
assistance systems, limiting the application of such methods in
autonomous driving. In this paper, we delve into the problem
of AR-MOT from the perspective of audio-video fusion and
audio-video tracking. We put forward EchoTrack, an end-to-
end AR-MOT framework with dual-stream vision transformers. The dual streams are intertwined with our Bidirectional
Frequency-domain Cross-attention Fusion Module (Bi-FCFM),
which bidirectionally fuses audio and video features from both
frequency- and spatiotemporal domains. Moreover, we propose
the Audio-visual Contrastive Tracking Learning (ACTL) regime
to extract homogeneous semantic features between expressions
and visual objects by learning homogeneous features between
different audio and video objects effectively. Aside from the
architectural design, we establish the first set of large-scale AR-
MOT benchmarks, including Echo-KITTI, Echo-KITTI+, and
Echo-BDD. Extensive experiments on the established benchmarks
demonstrate the effectiveness of the proposed EchoTrack model
and its components.



