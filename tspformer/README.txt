# TSPformer Description
PyTorch implementation of "Tspformer: Efficient Memory Transformer-based Network Model for TSP Combinatorial Optimization"

##Running Environment
GPU:NVIDIA GeForce RTX 2080 Ti
pytorch1.10
linux:ubuntu18.04
python3.9
matplotlib == 3.1
numpy == 1.19
pandas == 0.25
scikit_learn == 0.21

##Running Steps
training: python trainTspformer.py
testing:python testTspformer.py

##Parameters Setting
utils.options.get_options()

##References and Acknowledgements,taking the following repositories as baselines
1. https://github.com/xbresson/TSP_Transformer
2. https://github.com/zhouhaoyi/Informer2020
3. https://github.com/dreamgonfly/transformer-pytorch
4. https://github.com/MichelDeudon/encode-attend-navigate
5. https://github.com/wouterkool/attention-learn-to-route
6. https://github.com/pemami4911/neural-combinatorial-rl-pytorch


