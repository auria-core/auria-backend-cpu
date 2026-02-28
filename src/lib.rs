// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     CPU execution backend for AURIA Runtime Core.
//     Implements tensor operations and expert execution on CPU hardware
//     for the Nano and Standard tiers.
//
use auria_core::{ExecutionOutput, ExecutionState, AuriaResult, Tensor};
use async_trait::async_trait;

#[async_trait]
pub trait CpuBackend: Send + Sync {
    fn name(&self) -> &str;
    async fn execute(&self, input: Tensor, experts: Vec<Tensor>, state: ExecutionState) -> AuriaResult<ExecutionOutput>;
}

pub struct CpuExecutionEngine<B: CpuBackend> {
    backend: B,
}

impl<B: CpuBackend> CpuExecutionEngine<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub async fn execute(&self, input: Tensor, experts: Vec<Tensor>, state: ExecutionState) -> AuriaResult<ExecutionOutput> {
        self.backend.execute(input, experts, state).await
    }
}
