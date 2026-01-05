"""
SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot

This module implements the SparseGPT algorithm for pruning neural network layers.
Key concepts:
- Uses Optimal Brain Surgeon (OBS) framework
- Computes Hessian (H = X X^T) to understand weight importance
- Processes weights column-by-column, updating remaining weights to compensate for errors
- Supports both unstructured and N:M semi-structured sparsity
- Can be combined with quantization for joint compression
"""

import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

DEBUG = False 

# Disable TF32 for numerical precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class SparseGPT:
    """
    SparseGPT pruner for a single layer.
    
    The algorithm:
    1. Accumulate Hessian H = X X^T during calibration
    2. Compute H^(-1) using Cholesky decomposition
    3. Process columns left-to-right:
       a) Select mask based on saliency scores
       b) Prune/quantize weights
       c) Compute error
       d) Update remaining weights to compensate
    """

    def __init__(self, layer):
        """
        Initialize SparseGPT for a given layer.
        
        Args:
            layer: PyTorch layer (nn.Linear, nn.Conv2d, or transformers.Conv1D)
        """
        self.layer = layer
        self.dev = self.layer.weight.device
        
        # Get weight matrix and reshape if necessary
        W = layer.weight.data.clone()
        
        # Handle different layer types
        if isinstance(self.layer, nn.Conv2d):
            # Conv2d: (out_channels, in_channels, H, W) -> (out_channels, in_channels*H*W)
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            # Conv1D stores weights transposed
            W = W.t()
        
        # Store dimensions
        self.rows = W.shape[0]      # Number of output features
        self.columns = W.shape[1]   # Number of input features
        
        # Initialize Hessian matrix (will be accumulated during calibration)
        # H = X X^T where X is the input activation matrix
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        """
        Accumulate Hessian approximation H = X X^T from a batch of data.
        
        This function is called during the forward pass via hooks to collect
        input activations and accumulate the Hessian matrix.
        
        Args:
            inp: Input activations to the layer (batch_size, seq_len, features)
            out: Output activations (not used in Hessian computation)
            blocksize: Block size for processing (not used here)
            
        Example:
            If inp has shape (32, 2048, 4096):
            - 32 samples
            - 2048 sequence length
            - 4096 input features
            
            After reshaping: (32*2048, 4096) = (65536, 4096)
            Then transpose: (4096, 65536)
            H += (4096, 65536) @ (65536, 4096) = (4096, 4096)
        """
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
            
        # Ensure inp is 3D: (batch, seq_len, features)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        
        tmp = inp.shape[0]  # Batch size
        
        # Reshape input for Linear and Conv1D layers
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                # Flatten batch and sequence dimensions
                # (batch, seq_len, features) -> (batch*seq_len, features)
                inp = inp.reshape((-1, inp.shape[-1]))
            # Transpose to (features, samples)
            inp = inp.t()
        
        # Running average for numerical stability
        # New_H = (old_H * old_samples + new_contribution) / total_samples
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        
        # Scale input for numerical stability (Fisher information approximation)
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        
        # Accumulate H = X X^T
        # inp: (columns, samples)
        # inp.t(): (samples, columns)
        # Result: (columns, columns)
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        """
        Main pruning algorithm - processes weights column-by-column.
        
        Args:
            sparsity: Target sparsity (0.0 to 1.0). E.g., 0.5 = 50% of weights pruned
            prunen: N for N:M pruning (e.g., 2 for 2:4). 0 means unstructured
            prunem: M for N:M pruning (e.g., 4 for 2:4)
            blocksize: Process columns in blocks of this size (default 128)
            percdamp: Damping factor as percentage of mean diagonal (default 0.01)
            
        Example:
            sparsity=0.5, prunen=0, prunem=0: Prune 50% of weights (unstructured)
            sparsity=0, prunen=2, prunem=4: 2:4 structured sparsity
        """
        
        # ==================== STEP 1: PREPARE WEIGHT MATRIX ====================
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Setup quantizer if needed (for joint pruning + quantization)
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        # ==================== STEP 2: PREPARE HESSIAN ====================
        H = self.H
        del self.H  # Free memory
        
        # Handle dead neurons (features that never activate)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1  # Prevent division by zero
        W[:, dead] = 0     # Zero out weights for dead features

        Losses = torch.zeros(self.rows, device=self.dev)

        # Add damping for numerical stability
        # damp = percdamp * mean(diagonal of H)
        # This prevents issues when H is nearly singular
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        # ==================== STEP 3: COMPUTE HESSIAN INVERSE ====================
        # Use Cholesky decomposition for numerical stability
        # H = L L^T (Cholesky)
        # H^(-1) = (L L^T)^(-1) = (L^T)^(-1) L^(-1)
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H  # Now we have H^(-1)

        mask = None

        # ==================== STEP 4: PROCESS COLUMNS IN BLOCKS ====================
        # Process blocksize columns at a time (default 128)
        # This allows adaptive mask selection while maintaining efficiency
        
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            # Extract current block of weights
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)      # Will store pruned/quantized weights
            Err1 = torch.zeros_like(W1)    # Will store errors for propagation
            Losses1 = torch.zeros_like(W1) # Will store loss contributions
            Hinv1 = Hinv[i1:i2, i1:i2]     # Block of Hessian inverse

            # ==================== STEP 5: MASK SELECTION ====================
            # Two modes: unstructured (prunen=0) or N:M structured (prunen>0)
            
            if prunen == 0:  # Unstructured pruning
                if mask is not None:
                    # Use global mask (for special cases)
                    mask1 = mask[:, i1:i2]
                else:
                    # ========== SALIENCY SCORE CALCULATION ==========
                    # This is the OBS (Optimal Brain Surgeon) criterion
                    # Formula: saliency = w^2 / [H^(-1)]_ii
                    # 
                    # Intuition: 
                    # - Numerator (w^2): Larger weights are more important
                    # - Denominator ([H^(-1)]_ii): Considers second-order information
                    #   - Large [H^(-1)]_ii means this weight has little correlation with others
                    #   - Can be pruned with less impact on loss
                    #
                    # LOW saliency = SAFE to prune
                    # HIGH saliency = IMPORTANT, keep
                    
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    # tmp shape: (rows, count)
                    # Each element is the saliency score for that weight
                    
                    # Example:
                    # W1 = [[0.5, 0.1],    Hinv_diag = [0.1, 0.2]
                    #       [0.8, 0.2]]
                    # tmp = [[0.5^2/0.1^2, 0.1^2/0.2^2],  = [[25.0,  2.5],
                    #        [0.8^2/0.1^2, 0.2^2/0.2^2]]     [64.0,  1.0]]
                    # Low scores (2.5, 1.0) will be pruned first
                    
                    # Find threshold for target sparsity
                    # Sort all saliency scores and pick the value at sparsity percentile
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    
                    # ========== MASK CREATION ==========
                    # Create binary mask: True = prune (low saliency), False = keep
                    mask1 = tmp <= thresh
                    
                    # Example with 50% sparsity:
                    # tmp.flatten() = [25.0, 2.5, 64.0, 1.0]
                    # sorted = [1.0, 2.5, 25.0, 64.0]
                    # thresh = sorted[2] = 25.0 (at 50% position)
                    # mask1 = [[False, True],   (keep 25.0 and 64.0, prune 2.5 and 1.0)
                    #          [False, True]]
            else:
                # N:M structured sparsity - will be computed per column below
                mask1 = torch.zeros_like(W1) == 1  # All False initially

            # ==================== STEP 6: PROCESS EACH COLUMN ====================
            for i in range(count):
                w = W1[:, i]        # All rows, current column (shape: rows)
                d = Hinv1[i, i]     # Diagonal element of Hessian inverse (scalar)

                # ========== N:M STRUCTURED SPARSITY MASK SELECTION ==========
                if prunen != 0 and i % prunem == 0:
                    # For N:M pruning: every M consecutive weights, prune N of them
                    # Example: 2:4 means every 4 weights, prune 2 (50% sparsity in groups)
                    
                    # Compute saliency for next M weights
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    # tmp shape: (rows, M)
                    
                    # For each row, find the N weights with LOWEST saliency
                    # topk with largest=False gives us the N smallest values
                    indices = torch.topk(tmp, prunen, dim=1, largest=False)[1]
                    # indices shape: (rows, N)
                    
                    # Mark these N weights for pruning in the mask
                    mask1.scatter_(1, i + indices, True)
                    
                    # Example: 2:4 pruning, row 0
                    # W1[0, i:i+4] = [0.5, 0.8, 0.1, 0.3]
                    # tmp[0, :] = [saliency scores]
                    # Top 2 lowest: indices [2, 0] (positions of 0.1 and 0.5)
                    # mask1[0, i+2] = True, mask1[0, i+0] = True

                # ========== PRUNING ==========
                q = w.clone()
                q[mask1[:, i]] = 0  # Zero out pruned weights
                
                # Example:
                # w = [0.5, 0.8, 0.3, 0.9]
                # mask1[:, i] = [True, False, True, False]
                # q = [0.0, 0.8, 0.0, 0.9]  <- Pruned!

                # ========== QUANTIZATION (if enabled) ==========
                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), 
                        self.quantizer.scale, 
                        self.quantizer.zero, 
                        self.quantizer.maxq
                    ).flatten()
                    
                    # Example: 4-bit quantization
                    # q = [0.0, 0.8, 0.0, 0.9]
                    # After quantization: [0.0, 0.8, 0.0, 0.9] (rounded to nearest 4-bit value)

                Q1[:, i] = q
                
                # ========== ERROR CALCULATION ==========
                # Losses1 stores the squared error normalized by Hessian
                # This represents the contribution to the total loss from this column
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                
                # Example:
                # w = [0.5, 0.8, 0.3, 0.9]
                # q = [0.0, 0.8, 0.0, 0.9]
                # d = 0.1
                # Losses1[:, i] = [(0.5-0)^2/0.01, (0.8-0.8)^2/0.01, (0.3-0)^2/0.01, (0.9-0.9)^2/0.01]
                #               = [25.0, 0.0, 9.0, 0.0]

                # ========== WEIGHT UPDATE (Error Compensation) ==========
                # This is the KEY operation: update remaining unpruned weights
                # to compensate for the error introduced by pruning/quantization
                #
                # OBS Update Formula:
                # W_remaining -= (error / d) * H^(-1)[i, remaining_indices]
                #
                # Intuition:
                # - error = w - q: how much we changed this weight
                # - d = H^(-1)[i,i]: normalization factor
                # - H^(-1)[i, j]: correlation between weight i and weight j
                # - If weights are correlated, adjusting one affects the other
                
                err1 = (w - q) / d
                # err1 shape: (rows,)
                
                # Update all columns from i to end of block
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                #            ^                  ^                ^
                #            |                  |                |
                #         (rows, 1)         (rows, 1)      (1, count-i)
                #                              @
                #                          (rows, count-i)
                
                # Example:
                # err1 = [0.5, 0.0, 0.3, 0.0] / 0.1 = [5.0, 0.0, 3.0, 0.0]
                # Hinv1[i, i:] = [0.1, 0.05, 0.02, ...]
                # 
                # For row 0:
                # W1[0, i:] -= 5.0 * [0.1, 0.05, 0.02, ...]
                #           -= [0.5, 0.25, 0.1, ...]
                # 
                # This compensates for the error in column i by adjusting future columns!
                
                Err1[:, i] = err1

            # ==================== STEP 7: SAVE PRUNED WEIGHTS ====================
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            # ========== PROPAGATE ERRORS TO NEXT BLOCK ==========
            # Update weights in future blocks using accumulated errors
            # This ensures errors from this block are compensated in later blocks
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            # Err1: (rows, count) - errors from current block
            # Hinv[i1:i2, i2:]: (count, remaining_columns) - Hessian inverse cross terms
            # Result: (rows, remaining_columns) - adjustments for future blocks

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        # ==================== STEP 8: FINALIZE ====================
        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        # Reshape weights back to original layer format
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        """Free memory by deleting cached data."""
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

# ==================== ALGORITHM SUMMARY ====================
"""
SparseGPT Algorithm Flow:

1. CALIBRATION PHASE (add_batch):
   - Forward pass through model with calibration data
   - Accumulate Hessian H = X X^T for each layer
   - H captures correlations between input features

2. PRUNING PHASE (fasterprune):
   a) Prepare:
      - Compute H^(-1) using Cholesky decomposition
      - Add damping for numerical stability
   
   b) For each block of columns:
      - MASK SELECTION:
        * Compute saliency = w^2 / [H^(-1)]_ii
        * Prune weights with lowest saliency
      
      - For each column in block:
        * PRUNE: Zero out low-saliency weights
        * QUANTIZE (optional): Round to low precision
        * ERROR: Compute (original - pruned/quantized)
        * UPDATE: Adjust remaining weights using OBS formula
          W_remaining -= (error / d) * H^(-1)_cross_terms
      
      - PROPAGATE: Send errors to next block

3. FINALIZE:
   - Write pruned/quantized weights back to layer
   - Free memory

IMPORTANT POINTS:
- Saliency scoring: w^2 / [H^(-1)]_ii balances magnitude and Hessian info
- Column-wise processing: Enables reusing same H^(-1) for all rows
- Error compensation: Updates unpruned weights to minimize total loss
- Joint compression: Can combine pruning + quantization in single pass

COMPLEXITY:
- Hessian inverse: O(d^3) - computed once
- Pruning: O(d^2) per column, O(d^3) total
- Total: O(d^3) where d = hidden dimension
- Scales to 100B+ parameter models!
"""
