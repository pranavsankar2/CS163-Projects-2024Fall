---
layout: post
comments: true
title: Contrastive Language–Image Pre-training Applications and Extensions
author: UCLAdeepvision
date: 2024-12-13
---

> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance (https://arxiv.org/pdf/2407.05578v1)

### Background
FALIP is introduced in this paper as a novel method to enhance the zero-shot performance of the CLIP model. Its key strength lies in achieving this enhancement without modifying the original image or requiring additional training. While the baseline CLIP model already demonstrates impressive zero-shot performance across various tasks, previous methods have sought to improve it by employing visual prompts, such as colored circles or blur masks, to guide the model’s attention mechanisms. These visual prompts can effectively direct the CLIP model’s focus to specific regions of the image. However, they compromise the image’s overall integrity, creating a trade-off between preserving image fidelity and boosting model performance.

![YOLO]({{ '/assets/images/Team25/image2.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

### Methodology
The FALIP method for enhancing the CLIP model’s zero-shot performance involves multiple steps. First, the method applies a foveal attention mask to highlight the regions of attention (ROA) in the image. This mask is then integrated into CLIP’s multi-head self-attention module, which is used to conceptualize the relationships between various regions of the image. Finally, the model’s attention is aligned with human-like visual perception, focusing on specific regions without directly modifying the image.

![YOLO]({{ '/assets/images/Team25/image3.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

### Key Functional Features
The key functional distinctions between the CLIP model and the FALIP method proposed in this paper provide FALIP with a significant advantage over the standard CLIP pipeline. First, FALIP features a plug-and-play design, enabling the direct integration of CLIP models into the FALIP methodology without requiring substantial modifications to the model architecture. Additionally, the computational cost of this modified approach is considerably lower compared to other methods proposed for enhancing the CLIP model. Finally, preserving image integrity is a cornerstone of the FALIP method. It achieves this by guiding the model’s attention while maintaining the essential fidelity of the input image. This is accomplished through adaptive attention guidance, the core of the FALIP method, which inserts foveal attention masks into the multi-head self-attention module, enabling task-specific focus for the model.

### Applications
The FALIP protocol was evaluated on several zero-shot tasks, including Referring Expression Comprehension (REC), image classification, and 3D Point Cloud Recognition. For the Referring Expression Comprehension (REC) task, a five-stage process was employed to identify an image region based on a textual description. First, the input, consisting of an image with bounding boxes and a textual description, was processed through the pipeline, where the bounding boxes were transformed into masks. Next, the similarity between the text and the image regions was calculated. A “subtract” operation was then applied to reduce the weights of less relevant similarities. Finally, the best-matching region was selected based on the computed scores. The Image Classification task followed a slightly modified process. In this case, the bounding boxes were converted into a single mask. The protocol then calculated similarity scores by comparing the image to each category’s textual description to determine the best match. For 3D Point Cloud Recognition, the 3D point cloud was projected into six 2D depth maps. The foreground portions of these depth maps were converted into masks, and similarity scores were computed between each view and the category texts. The views were then weighted and combined to produce the final prediction. By evaluating the FALIP protocol across these zero-shot tasks, its effectiveness and versatility in diverse applications were demonstrated.

![YOLO]({{ '/assets/images/Team25/image4.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

![YOLO]({{ '/assets/images/Team25/image5.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

![YOLO]({{ '/assets/images/Team25/image1.jpg' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

### Results and Discussion
The FALIP protocol, as outlined in this paper, has demonstrated competitive results across numerous datasets for tested zero-shot tasks, underscoring its efficacy as a novel and improved method for zero-shot learning with CLIP. At its core, the modification of attention head sensitivity forms the foundation of the proposed technology. This approach was motivated by the discovery that CLIP models exhibit significant variability in their responses to visual prompts and that the attention heads in CLIP possess differing levels of sensitivity to the visual cues provided as input. This variability in sensitivity, which serves as a tunable hyperparameter, offers opportunities for further improvement by enabling testing and adjustment to enhance the effectiveness of visual prompts. Moreover, FALIP’s adaptability for domain-specific problems represents a critical area of interest for researchers and practitioners looking to apply it to other fields of study or industry applications. Overall, these results and insights highlight FALIP’s potential as a powerful and flexible approach for enhancing CLIP’s zero-shot capabilities across various visual understanding tasks, while also paving the way for future advancements in vision-language models.










## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
