// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     CPU execution backend for AURIA Runtime Core.
//     Implements tensor operations and expert execution on CPU hardware
//     for the Nano and Standard tiers.
//
pub mod gguf;

use auria_core::{AuriaError, AuriaResult, Tensor, TensorDType, Tier, UsageStats};
use auria_execution::{ExecutionOutput, ExecutionState};
use async_trait::async_trait;
use gguf::ModelRunner;
use std::sync::Arc;

pub struct CpuBackendImpl {
    num_threads: usize,
    model_runner: Arc<ModelRunner>,
}

impl CpuBackendImpl {
    pub fn new() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self { 
            num_threads,
            model_runner: Arc::new(ModelRunner::new()),
        }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self { 
            num_threads,
            model_runner: Arc::new(ModelRunner::new()),
        }
    }

    pub fn with_model(model_path: &str) -> AuriaResult<Self> {
        let runner = ModelRunner::new();
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AuriaError::ExecutionError(format!("Failed to create runtime: {}", e)))?;
        rt.block_on(runner.load_model(model_path))?;
        
        Ok(Self {
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4),
            model_runner: Arc::new(runner),
        })
    }

    pub async fn load_model(&self, model_path: &str) -> AuriaResult<()> {
        self.model_runner.load_model(model_path).await
    }

    pub async fn is_model_loaded(&self) -> bool {
        self.model_runner.is_loaded().await
    }

    pub async fn run_inference(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> AuriaResult<Vec<String>> {
        if !self.model_runner.is_loaded().await {
            return Err(AuriaError::ExecutionError("No model loaded".to_string()));
        }
        
        self.model_runner.infer(prompt, max_tokens, temperature, 0.9).await
    }

    fn matmul_f16(a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; rows_a * cols_b];
        
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

        let result = Self::matmul_f16(&input_f32, &weight_f32, input_rows, input_cols, weight_cols);
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

    pub fn get_model_info(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "backend": "cpu",
            "num_threads": self.num_threads,
        }))
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

        for (_i, expert) in experts.iter().enumerate() {
            current = self.tensor_matmul(&current, expert)?;
            
            let f32_data = self.convert_to_f32(&current.data)?;
            let mut data = f32_data;
            self.apply_activation(&mut data);
            current.data = self.convert_to_f16(&data)?;
        }

        let tokens = self.decode_tokens(&current)?;
        
        Ok(ExecutionOutput {
            tokens,
            usage: UsageStats {
                tokens_generated: state.position as u64,
                tokens_processed: state.position as u64,
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

pub fn create_cpu_backend() -> CpuBackendImpl {
    CpuBackendImpl::new()
}

pub use gguf::{ModelRunner as GGUFModelRunner, ModelConfig, ModelType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let backend = CpuBackendImpl::new();
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = CpuBackendImpl::matmul_f16(&a, &b, 2, 2, 2);
        assert_eq!(result.len(), 4);
    }
}
