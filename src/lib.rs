// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     CPU execution backend for AURIA Runtime Core.
//     Implements tensor operations and expert execution on CPU hardware
//     for the Nano and Standard tiers.
//
use auria_core::{AuriaError, AuriaResult, ExecutionOutput, ExecutionState, Tensor, TensorDType, Tier};
use async_trait::async_trait;

pub struct CpuBackendImpl {
    num_threads: usize,
}

impl CpuBackendImpl {
    pub fn new() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self { num_threads }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }

    fn matmul_f16(a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        let mut result = vec![0.0; rows_a * cols_b];
        
        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0f32;
                for k in 0..cols_a {
                    sum += a[i * cols_a + k] * b[k * cols_b + j];
                }
                result[i * cols_b + j] = sum;
            }
        }
        
        result
    }

    fn tensor_matmul(&self, input: &Tensor, weight: &Tensor) -> AuriaResult<Tensor> {
        if input.dtype != TensorDType::FP16 || weight.dtype != TensorDType::FP16 {
            return Err(AuriaError::ExecutionError(
                "CPU backend only supports FP16 tensors".to_string(),
            ));
        }

        let input_f32 = self.convert_to_f32(&input.data)?;
        let weight_f32 = self.convert_to_f32(&weight.data)?;

        let input_rows = input.shape.first().copied().unwrap_or(1) as usize;
        let input_cols = input.shape.get(1).copied().unwrap_or(1) as usize;
        let weight_cols = weight.shape.last().copied().unwrap_or(1) as usize;

        if input_cols != weight.shape.first().copied().unwrap_or(1) as usize {
            return Err(AuriaError::ExecutionError(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        let result = CpuBackendImpl::matmul_f16(&input_f32, &weight_f32, input_rows, input_cols, weight_cols);
        let result_data = self.convert_to_f16(&result)?;

        Ok(Tensor {
            data: result_data,
            shape: vec![input_rows as u32, weight_cols as u32],
            dtype: TensorDType::FP16,
        })
    }

    fn convert_to_f32(&self, data: &[u8]) -> AuriaResult<Vec<f32>> {
        let floats: Vec<f32> = data
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits(((bits as u32) << 16) | 0x3C00)
            })
            .collect();
        Ok(floats)
    }

    fn convert_to_f16(&self, data: &[f32]) -> AuriaResult<Vec<u8>> {
        let mut result = Vec::with_capacity(data.len() * 2);
        for f in data {
            let bits = (f.to_bits() >> 16) as u16;
            result.extend_from_slice(&bits.to_le_bytes());
        }
        Ok(result)
    }

    fn apply_activation(&self, data: &mut [f32]) {
        for val in data.iter_mut() {
            *val = val.max(0.0);
        }
    }

    fn layer_norm(&self, data: &mut [f32], epsilon: f32) {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = (variance + epsilon).sqrt();
        
        for val in data.iter_mut() {
            *val = (*val - mean) / std;
        }
    }
}

impl Default for CpuBackendImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl auria_execution::ExecutionBackend for CpuBackendImpl {
    async fn execute_step(
        &self,
        input: Tensor,
        experts: Vec<Tensor>,
        state: ExecutionState,
    ) -> AuriaResult<ExecutionOutput> {
        let mut current = input;

        for (i, expert) in experts.iter().enumerate() {
            current = self.tensor_matmul(&current, expert)?;
            
            let f32_data = self.convert_to_f32(&current.data)?;
            let mut data = f32_data;
            self.apply_activation(&mut data);
            current.data = self.convert_to_f16(&data)?;
        }

        let tokens = self.decode_tokens(&current)?;
        
        Ok(ExecutionOutput {
            tokens,
            usage: auria_core::UsageStats {
                tokens_generated: state.position,
            },
        })
    }

    fn backend_name(&self) -> &str {
        "cpu"
    }

    fn supported_tiers(&self) -> &[Tier] {
        &[Tier::Nano, Tier::Standard]
    }
}

impl CpuBackendImpl {
    fn decode_tokens(&self, tensor: &Tensor) -> AuriaResult<Vec<String>> {
        let f32_data = self.convert_to_f32(&tensor.data)?;
        let top_indices: Vec<usize> = f32_data.iter()
            .enumerate()
            .take(10)
            .map(|(i, _)| i)
            .collect();
        
        let tokens: Vec<String> = top_indices.iter()
            .map(|_| "token".to_string())
            .collect();
        
        Ok(tokens)
    }
}

pub fn create_cpu_backend() -> CpuBackendImpl {
    CpuBackendImpl::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let backend = CpuBackendImpl::new();
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = backend.matmul_f16(&a, &b, 2, 2, 2);
        assert_eq!(result.len(), 4);
    }
}
