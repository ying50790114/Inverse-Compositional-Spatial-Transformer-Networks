# Implementation of 'Inverse Compositional Spatial Transformer Networks'

---
This is a TensorFlow2 implementation of the method described in the paper 'Inverse Compositional Spatial Transformer Networks' by Chen-Hsuan Lin, and Simon Lucey.

**Paper Link:** https://arxiv.org/pdf/1612.03897

**Dataset:** 利用 MNIST 手寫數字數據集實現本篇論文。

**Dependencies**
```
Python: 3.7
TensorFlow: 2.5.1
```


**Method:**
- 建立Lucas & Kanade (LK)算法和空間變換網絡 (STNs) 之間的理論聯繫，並受LK算法逆合成 (IC) 變體的啟發。
- 在IC-STNs框架內使用幾何預測器和遞歸空間變換。
- Skill: Spatial Transformer Networks、Lucas-Kanade Algorithm、Affine Transformation

**network.py:** 參考原論文，並稍微調整一些參數。

**warp.py:** 在此採用affine transformation。

**load_data.py:** 

**main.py:** 透過辨識錯誤優化模型參數。


---

### Acknowledgments
The code is basically a modification of [inverse-compositional-STN](https://github.com/chenhsuanlin/inverse-compositional-STN) implemented in TensorFlow 2. All credit goes to the authors of [IC-STN](https://arxiv.org/abs/1612.03897), Chen-Hsuan Lin and Simon Lucey.

[[Paper]](https://arxiv.org/abs/1612.03897) 
[[Code(PyTorch)]](https://github.com/chenhsuanlin/inverse-compositional-STN/tree/master/MNIST-pytorch) 
[[Code(TensorFlow1)]](https://github.com/chenhsuanlin/inverse-compositional-STN/tree/master/MNIST-tensorflow)
```
@inproceedings{lin2017inverse,
  title={Inverse Compositional Spatial Transformer Networks},
  author={Lin, Chen-Hsuan and Lucey, Simon},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2017}
}
```


---
### References
- https://github.com/chenhsuanlin/inverse-compositional-STN

