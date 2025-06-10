# Document

## Models Overview

This section provides a brief introduction to the segmentation models integrated in our project. The models are based on a wide range of architectures including **CNNs**, **enhanced UNet variants**, **Vision Transformers**, and **State Space Models**, enabling a comprehensive evaluation of their performance on medical image segmentation tasks.

- **UNet**  
  The classical encoder-decoder architecture with skip connections, originally designed for biomedical image segmentation. Known for its simplicity and strong performance on small datasets.

- **VMUNet**  
  Vision Mamba-based UNet variant that incorporates a visual state space model into the UNet architecture to better model long-range dependencies while maintaining efficient computation.

- **U2Net**  
  Deep two-level nested U-shaped network that leverages residual U-blocks (RSUs) to capture multi-scale contextual information, achieving fine-grained segmentation outputs.

- **UNet++**  
  Enhanced UNet architecture with nested and dense skip pathways, allowing better feature fusion and multi-scale representation across the network.

- **UNet+++**  
  An advanced extension of UNet++ with further optimized skip connection strategies and enhanced decoder design, aimed at improving segmentation accuracy and feature expressiveness.

- **VMUNetV2**  
  A second-generation Vision Mamba UNet architecture that introduces refined visual state space modules and improved visual token mixing, designed for higher stability and segmentation accuracy.

- **HVMUNet**  
  Hybrid Vision Mamba UNet that combines convolutional and state space modeling techniques, offering strong performance on heterogeneous and non-IID medical imaging datasets.

- **TransUNet**  
  Transformer-based UNet architecture that integrates a Vision Transformer (ViT) encoder with a UNet-style decoder, effectively capturing both global and local features.

- **ResUNet**  
  A UNet variant with residual connections added to the encoder and decoder blocks, facilitating gradient flow and enhancing learning stability.

- **ResUNet++**  
  An improved version of ResUNet that incorporates atrous spatial pyramid pooling (ASPP), squeeze-and-excitation modules, and redesigned residual blocks for richer contextual understanding.