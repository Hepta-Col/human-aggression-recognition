# Violence Datasets

## Overview

| Name                     | Scale                  | Length per clip (sec) | Resolution | Annotation  | Scenario     |
| ------------------------ | ---------------------- | --------------------- | ---------- | ----------- | ------------ |
| BEHAVE                   | 4 videos (171 clips)   | 0.24~61.92            | 640x480    | frame-level | acted fights |
| RE-DID                   | 30 videos              | 20~240                | 1280x720   | frame-level | natural      |
| VSD                      | 18 movies (1317 clips) | 55.3~829.4            | variable   | frame-level | movies       |
| CCTV-Fights              | 1000 clips             | 5~720                 | variable   | frame-level | surveillance, mobile cameras |
| Hockey Fight             | 1000 clips             | 1.6~1.96              | 360x288    | video-level | Hockey       |
| Movies Fight             | 200 clips              | 1.6~2                 | 720x480    | video-level | movies, sports |
| Crowd Violence           | 246 clips              | 1.04~6.52             | variable   | video-level | natural      |
| SBU Kinect Interaction   | 264 lips               | 0.67~3                | 640x480    | video-level | acted fights |
| Violent-Flows            | 246 clips              | 1.04~6.52            | 320x240    | video-level | streets/school/sports|
| Avenue | 37 videos |  |  | only normal  videos are present in training set |  |
| ——————— | ——————— | —————— | —————— | —————— | ——————— |
| UCF-Crime            | 1900 clips             | 60~600                | variable   | video-level | surveillance |
| UCFCrime2Local           |                        |                       |            | frame-level |  |
| UCF-Crime annotated      |                        |                       |            | frame-level |              |
| **RWF-2000**             | 2000 clips             | 5                     | variable   | video-level | surveillance |
| **fight-detection-surv** | 300 videos             | 2                     | variable   | video-level | surveillance |
| **RLV**                  | 2000 clips             | 3-7                   | 397x511(?) | video-level | natural (?)  |
| XD-Violence              | 4754 videos            | 1~240 (mostly) | variable  | video-level, multiple label | movies, sports, games, hand-held cameras, surveillance, car cameras, etc… |
| Shanghai-Tech            | 437 videos |                       |            |             | surveillance |
| UCSD | 98 videos | | | frame-level | surveillance |
| ——————— | ——————— | —————— | —————— | —————— | ——————— |
| **YouTube-Small** (ours) | 58 - 41 clips(fight/non-fight) | 2~3                 | variable   | video-level | natural      |

* we have the bold ones on HAL  

## UCF-Crime

> paper link: https://arxiv.org/pdf/1801.04264.pdf
>
> webpage: https://webpages.charlotte.edu/cchen62/dataset.html

### Details

1. contains 1,900 untrimmed real-world street and indoor surveillance videos with a total duration of 128 hours.
2. the training set contains 1,610 videos with video-level labels, and the test set contains 290 videos with frame-level labels.
3. 13 realistic anomalies: Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism. 

### Leader Board

| Model                                                        | Reported on Conference/Journal | Supervised | Feature       | Encoder-based | 32 Segments | AUC (%) | FAR@0.5 on Normal (%) |
| ------------------------------------------------------------ | ------------------------------ | ---------- | ------------- | ------------- | ----------- | ------- | --------------------- |
| [ST-Graph](https://github.com/fjchange/awesome-video-anomaly-detection#02014) | ACM MM 20                      | Un         | -             | √             | X           | 72.7    |                       |
| [Sultani.etl](https://github.com/fjchange/awesome-video-anomaly-detection#11801) | CVPR 18                        | Weakly     | C3D RGB       | X             | √           | 75.41   | 1.9                   |
| [IBL](https://github.com/fjchange/awesome-video-anomaly-detection#11903) | ICIP 19                        | Weakly     | C3D RGB       | X             | √           | 78.66   | -                     |
| [Motion-Aware](https://github.com/fjchange/awesome-video-anomaly-detection#11904) | BMVC 19                        | Weakly     | PWC Flow      | X             | √           | 79.0    | -                     |
| [Background-Bias](https://github.com/fjchange/awesome-video-anomaly-detection#21901) | ACM MM 19                      | Fully      | NLN RGB       | √             | X           | 82.0    | -                     |
| [GCN-Anomaly](https://github.com/fjchange/awesome-video-anomaly-detection#11901) | CVPR 19                        | Weakly     | TSN RGB       | √             | X           | 82.12   | 0.1                   |
| [MIST](https://github.com/fjchange/awesome-video-anomaly-detection#12101) | CVPR 21                        | Weakly     | I3D RGB       | √             | X           | 82.30   | 0.13                  |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12201) | AAAI 22                        | Weakly     | C3D RGB       | √             | X           | 82.85   | -                     |
| [CLAWS](https://github.com/fjchange/awesome-video-anomaly-detection#12004) | ECCV 20                        | Weakly     | C3D RGB       | √             | X           | 83.03   | -                     |
| [RTFM](https://github.com/fjchange/awesome-video-anomaly-detection#12102) | ICCV 21                        | Weakly     | I3D RGB       | X             | √           | 84.03   | -                     |
| [CRFD](https://github.com/fjchange/awesome-video-anomaly-detection#12105) | TIP 21                         | Weakly     | I3D RGB       | X             | √           | 84.89   | -                     |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12202) | AAAI 22                        | Weakly     | I3D RGB       | √             | X           | 85.30   | -                     |
| [WSAL](https://github.com/fjchange/awesome-video-anomaly-detection#12104) | TIP 21                         | Weakly     | I3D RGB       | X             | √           | 85.38   | -                     |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12201) | AAAI 22                        | Weakly     | VideoSwin-RGB | √             | X           | 85.62   | -                     |

## RWF-2000

> paper link: https://arxiv.org/pdf/1911.05913.pdf
>
> webpage: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

### Details

1. contains 2,000 videos captured by surveillance cameras in real-world scenes.

### Leader Board



## XD-Violence

> paper link: https://arxiv.org/pdf/2007.04687.pdf
>
> webpage: https://roc-ng.github.io/XD-Violence/

### Details

1. contains 4,754 untrimmed videos (2405 violent and 2349 non-violent) with a total duration of 217 hours.
2. 6 physically violent classes: abuse, car accident, explosion, fighting, riot, shooting
3. collect from multiple sources, such as movies, sports, surveillances, and CCTVs, with **audio signals**.
4. the training set contains 3,954 videos with **video-level** labels, and the test set contains 800 videos with **frame-level** labels.
5. multiple violent labels (1~3) for each violent video.

### Leader Board

| Model                                                        | Reported on Conference/Journal | Supervision | Feature       | Encoder-based | 32 Segments | AP(%) |
| ------------------------------------------------------------ | ------------------------------ | ----------- | ------------- | ------------- | ----------- | ----- |
| [Wu et al.](https://github.com/fjchange/awesome-video-anomaly-detection#12003) | ECCV 2020                      | Weakly      | C3D-RGB       | X             | X           | 67.19 |
| [Sultani et al.](https://github.com/fjchange/awesome-video-anomaly-detection#11801) | ECCV 2020 (reported by Wu)     | Weakly      | I3D-RGB       | X             | √           | 73.20 |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12201) | AAAI 2022                      | Weakly      | C3D-RGB       | X             | X           | 75.53 |
| [CRFD](https://github.com/fjchange/awesome-video-anomaly-detection#12105) | TIP 2021                       | Weakly      | I3D-RGB       | X             | √           | 75.90 |
| [RTFM](https://github.com/fjchange/awesome-video-anomaly-detection#12102) | ICCV 2021                      | Weakly      | I3D-RGB       | X             | √           | 77.81 |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12201) | AAAI 2022                      | Weakly      | I3D-RGB       | X             | X           | 78.28 |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12201) | AAAI 2022                      | Weakly      | VideoSwin-RGB | X             | X           | 78.59 |
| [Wu et al.](https://github.com/fjchange/awesome-video-anomaly-detection#12003) | ECCV 2020                      | Weakly      | I3D-RGB+Audio | X             | X           | 78.64 |

## Shanghai-Tech

> paper link: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf
>
> webpage: https://svip-lab.github.io/dataset/campus_dataset.html; https://github.com/desenzhou/ShanghaiTechDataset

### Details

1. contains 437 campus surveillance videos with 130 abnormal events in 13 scenes.
2. all the videos in the training set are normal.

### Leader Board

| Model                                                        | Reported on Conference/Journal | Supervision                   | Feature            | Encoder-based | AUC(%) | FAR@0.5 (%) |
| ------------------------------------------------------------ | ------------------------------ | ----------------------------- | ------------------ | ------------- | ------ | ----------- |
| [Conv-AE](https://github.com/fjchange/awesome-video-anomaly-detection#01601) | CVPR 16                        | Un                            | -                  | √             | 60.85  | -           |
| [stacked-RNN](https://github.com/fjchange/awesome-video-anomaly-detection#01702) | ICCV 17                        | Un                            | -                  | √             | 68.0   | -           |
| [MNAD](https://github.com/fjchange/awesome-video-anomaly-detection#02005) | CVPR 20                        | Un                            | -                  | √             | 70.5   | -           |
| [Mem-AE](https://github.com/fjchange/awesome-video-anomaly-detection#01901) | ICCV 19                        | Un                            | -                  | √             | 71.2   | -           |
| [FramePred](https://github.com/fjchange/awesome-video-anomaly-detection#01801) | CVPR 18                        | Un                            | -                  | √             | 72.8   | -           |
| [FramePred*](https://github.com/fjchange/awesome-video-anomaly-detection#11902) | IJCAI 19                       | Un                            | -                  | √             | 73.4   | -           |
| [AMMC](https://github.com/fjchange/awesome-video-anomaly-detection#02101) | AAAI 21                        | Un                            | -                  | √             | 73.7   | -           |
| [ST-Graph](https://github.com/fjchange/awesome-video-anomaly-detection#02014) | ACM MM 20                      | Un                            | -                  | √             | 74.7   | -           |
| [VEC](https://github.com/fjchange/awesome-video-anomaly-detection#02011) | ACM MM 20                      | Un                            | -                  | √             | 74.8   | -           |
| [MLEP](https://github.com/fjchange/awesome-video-anomaly-detection#11902) | IJCAI 19                       | 10% test vids with Video Anno | -                  | √             | 75.6   | -           |
| [HF2-VAD](https://github.com/fjchange/awesome-video-anomaly-detection#02103) | ICCV 21                        | Un                            | -                  | √             | 76.2   | -           |
| [GCN-Anomaly](https://github.com/fjchange/awesome-video-anomaly-detection#11901) | CVPR 19                        | Weakly (Re-Organized Dataset) | C3D-RGB            | √             | 76.44  | -           |
| [ROADMAP](https://github.com/fjchange/awesome-video-anomaly-detection#02104) | TNNLS 21                       | Un                            | -                  | √             | 76.6   | -           |
| [MLEP](https://github.com/fjchange/awesome-video-anomaly-detection#11902) | IJCAI 19                       | 10% test vids with Frame Anno | -                  | √             | 76.8   | -           |
| [BDPN](https://github.com/fjchange/awesome-video-anomaly-detection#02202) | AAAI 22                        | Un                            | -                  | √             | 78.1   | -           |
| [CAC](https://github.com/fjchange/awesome-video-anomaly-detection#02013) | ACM MM 20                      | Un                            | -                  | √             | 79.3   |             |
| [IBL](https://github.com/fjchange/awesome-video-anomaly-detection#12002) | ICME 2020                      | Weakly (Re-Organized Dataset) | I3D-RGB            | X             | 82.5   | 0.10        |
| [GCN-Anomaly](https://github.com/fjchange/awesome-video-anomaly-detection#11901) | CVPR 19                        | Weakly (Re-Organized Dataset) | TSN-Flow           | √             | 84.13  | -           |
| [GCN-Anomaly](https://github.com/fjchange/awesome-video-anomaly-detection#11901) | CVPR 19                        | Weakly (Re-Organized Dataset) | TSN-RGB            | √             | 84.44  | -           |
| [Sultani.etl](https://github.com/fjchange/awesome-video-anomaly-detection#12002) | ICME 2020                      | Weakly (Re-Organized Dataset) | C3D-RGB            | X             | 86.3   | 0.15        |
| [CLAWS](https://github.com/fjchange/awesome-video-anomaly-detection#12004) | ECCV 20                        | Weakly (Re-Organized Dataset) | C3D-RGB            | √             | 89.67  |             |
| [SSMT](https://github.com/fjchange/awesome-video-anomaly-detection#02102) | CVPR 21                        | Un                            | -                  | √             | 90.2   | -           |
| [AR-Net](https://github.com/fjchange/awesome-video-anomaly-detection#12002) | ICME 20                        | Weakly (Re-Organized Dataset) | I3D-RGB & I3D Flow | X             | 91.24  | 0.10        |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12201) | AAAI 22                        | Weakly (Re-Organized Dataset) | C3D-RGB            | X             | 94.81  | -           |
| [MIST](https://github.com/fjchange/awesome-video-anomaly-detection#12101) | CVPR 21                        | Weakly (Re-Organized Dataset) | I3D-RGB            | √             | 94.83  | 0.05        |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12201) | AAAI 22                        | Weakly (Re-Organized Dataset) | I3D-RGB            | X             | 96.08  | -           |
| [RTFM](https://github.com/fjchange/awesome-video-anomaly-detection#12102) | ICCV 21                        | Weakly (Re-Organized Dataset) | I3D-RGB            | X             | 97.21  | -           |
| [MSL](https://github.com/fjchange/awesome-video-anomaly-detection#12201) | AAAI 22                        | Weakly (Re-Organized Dataset) | VideoSwin-RGB      | X             | 97.32  | -           |
| [CRFD](https://github.com/fjchange/awesome-video-anomaly-detection#12105) | TIP 21                         | Weakly (Re-Organized Dataset) | I3D-RGB            | X             | 97.48  | -           |

## Avenue

>paper link: http://www.cse.cuhk.edu.hk/leojia/papers/abnormaldect_iccv13.pdf
>
>webpage: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

### Details

1. contains 16 training and 21 testing video clips. The videos are captured in CUHK campus avenue with 30652 (15328 training, 15324 testing) frames in total.
2. The training videos capture normal situations. Testing videos include both normal and abnormal events.

### Leader Board

| Model                                                        | Reported on Conference/Journal | Supervision                   | Feature                | End2End | AUC(%) |
| ------------------------------------------------------------ | ------------------------------ | ----------------------------- | ---------------------- | ------- | ------ |
| [Conv-AE](https://github.com/fjchange/awesome-video-anomaly-detection#01601) | CVPR 16                        | Un                            | -                      | √       | 70.2   |
| [ConvLSTM-AE](https://github.com/fjchange/awesome-video-anomaly-detection#01703) | ICME 17                        | Un                            | -                      | √       | 77.0   |
| [Conv-AE*](https://github.com/fjchange/awesome-video-anomaly-detection#01801) | CVPR 18                        | Un                            | -                      | √       | 80.0   |
| [Unmasking](https://github.com/fjchange/awesome-video-anomaly-detection#01705) | ICCV 17                        | Un                            | 3D gradients+VGG conv5 | X       | 80.6   |
| [stacked-RNN](https://github.com/fjchange/awesome-video-anomaly-detection#01702) | ICCV 17                        | Un                            | -                      | √       | 81.7   |
| [Mem-AE](https://github.com/fjchange/awesome-video-anomaly-detection#01901) | ICCV 19                        | Un                            | -                      | √       | 83.3   |
| [DeepAppearance](https://github.com/fjchange/awesome-video-anomaly-detection#01706) | ICAIP 17                       | Un                            | -                      | √       | 84.6   |
| [FramePred](https://github.com/fjchange/awesome-video-anomaly-detection#01801) | CVPR 18                        | Un                            | -                      | √       | 85.1   |
| [AMMC](https://github.com/fjchange/awesome-video-anomaly-detection#02101) | AAAI 21                        | Un                            | -                      | √       | 86.6   |
| [Appearance-Motion Correspondence](https://github.com/fjchange/awesome-video-anomaly-detection#01904) | ICCV 19                        | Un                            | -                      | √       | 86.9   |
| [CAC](https://github.com/fjchange/awesome-video-anomaly-detection#02013) | ACM MM 20                      | Un                            | -                      | √       | 87.0   |
| [ROADMAP](https://github.com/fjchange/awesome-video-anomaly-detection#02104) | TNNLS 21                       | Un                            | -                      | √       | 88.3   |
| [MNAD](https://github.com/fjchange/awesome-video-anomaly-detection#02005) | CVPR 20                        | Un                            | -                      | √       | 88.5   |
| [FramePred*](https://github.com/fjchange/awesome-video-anomaly-detection#11902) | IJCAI 19                       | Un                            | -                      | √       | 89.2   |
| [ST-Graph](https://github.com/fjchange/awesome-video-anomaly-detection#02014) | ACM MM 20                      | Un                            | -                      | √       | 89.6   |
| [VEC](https://github.com/fjchange/awesome-video-anomaly-detection#02011) | ACM MM 20                      | Un                            | -                      | √       | 90.2   |
| [AEP](https://github.com/fjchange/awesome-video-anomaly-detection#02105) | TNNLS 21                       | Un                            | -                      | √       | 90.2   |
| [Causal](https://github.com/fjchange/awesome-video-anomaly-detection#02201) | AAAI 22                        | Un                            | I3D-RGB                | X       | 90.3   |
| [BDPN](https://github.com/fjchange/awesome-video-anomaly-detection#02202) | AAAI 22                        | Un                            | -                      | √       | 90.3   |
| [HF2-VAD](https://github.com/fjchange/awesome-video-anomaly-detection#02103) | ICCV 21                        | Un                            | -                      | √       | 91.1   |
| [MLEP](https://github.com/fjchange/awesome-video-anomaly-detection#11902) | IJCAI 19                       | 10% test vids with Video Anno | -                      | √       | 91.3   |
| [SSMT](https://github.com/fjchange/awesome-video-anomaly-detection#02102) | CVPR 21                        | Un                            | -                      | √       | 92.8   |
| [MLEP](https://github.com/fjchange/awesome-video-anomaly-detection#11902) | IJCAI 19                       | 10% test vids with Frame Anno | -                      | √       | 92.8   |

## UCSD

> paper link: 
>
> webpage: http://www.svcl.ucsd.edu/projects/anomaly/dataset.html

### Details

1. acquired with a stationary camera mounted at an elevation, overlooking pedestrian walkways. 

2. the crowd density in the walkways was variable, ranging from sparse to very crowded. 

3. in the normal setting, the video contains only pedestrians. Abnormal events are due to either: the circulation of non pedestrian entities in the walkways; or anomalous pedestrian motion patterns.

4. Commonly occurring anomalies include bikers, skaters, small carts, and people walking across a walkway or in the grass that surrounds it. A few instances of people in wheelchair were also recorded. 

5. The data was split into 2 subsets, each corresponding to a different scene. The video footage recorded from each scene was split into various clips of around 200 frames.

   **Peds1:** clips of groups of people walking towards and away from the camera, and some amount of perspective distortion. Contains 34 training video samples and 36 testing video samples.

   **Peds2:** scenes with pedestrian movement parallel to the camera plane. Contains 16 training video samples and 12 testing video samples.

6. For each clip, the ground truth annotation includes a binary flag per frame, indicating whether an anomaly is present at that frame. In addition, a subset of 10 clips for Peds1 and 12 clips for Peds2 are provided with manually generated pixel-level binary masks, which identify the regions containing anomalies. 

### Leader Board

